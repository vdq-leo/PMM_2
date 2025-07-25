from algo_deploy.strategy_module import LiveStrategy, Indicator, extract_ochlv
from typing import Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
import time
import numpy as np

class PMM002(LiveStrategy):
    """
    PMM0002: MACD adjusted price + Bollinger bands quoting + Inventoryt Skew
    """

    def __init__(self, strategy_name: str, params: Dict, account_id: str, exchange: str, base: str, quote: str,
                 category: str = 'futures', demo: bool = True):
        super().__init__(strategy_name, account_id, exchange, base, quote, category, demo)
        self.add_params(params)
        self.stop_flag = False

        self.executor = ThreadPoolExecutor(max_workers=2)
        # Setup signal handler once on init
        signal.signal(signal.SIGINT, self._signal_handler)

        self.init_balance: float = 0

    def add_params(self, params: dict):
        """Update strategy parameters."""
        self.period = params.get('period', 14)
        self.alpha = params.get('alpha', 0.1)
        self.max_position = params.get('max_position', 5)
        self.value_entry = params.get('value_entry', 10)
        self.sleep_time = params.get('sleep_time', 60)
        self.stop_loss_threshold = params.get('stop_loss_threshold', 0.2)

    def get_opened_positions(self) -> dict:
        """Get opened positions."""
        open_positions = self.client.get_open_positions(self.base).copy()['data'][0]
        open_size = open_positions.get('size', 0)
        side = open_positions.get('side', '')
        
        if side!= '':
            sign = 1 if side.upper() == 'BUY' else -1
        else:
            return 0
        
        pos_norm = float(open_size) / float(self.value_entry) * sign
        return int(pos_norm)

    def get_entry_price(self) -> tuple[float, float]:
        """Returns fair quote price with position skew and adaptive half-spread"""
        try:
            # 5m candles
            candles = self.get_candles(self.base, self.quote, interval="5m")
            o, c, h, l, v = extract_ochlv(candles)
            macd = Indicator.macd(c, fast_period=12, slow_period=26, signal_period=9)[-1]
            _, trend = Indicator.trend_detect(h, l, c, self.period)
            upper, mid, _ = Indicator.bollinger_bands(c, self.period, self.alpha)

            # 1h candles
            candles_ = self.get_candles(self.base, self.quote, interval="1h")
            o_, c_, h_, l_, v_ = extract_ochlv(candles_)
            macd_ = Indicator.macd(c_, fast_period=12, slow_period=26, signal_period=9)[-1]
            atr_ = Indicator.average_true_range(h_, l_, c_, self.period)

            in_trend = trend[-1] == 1
            half_spread = abs(upper[-1] - mid[-1])
            if in_trend:
                half_spread *= 2  # widen if strong trend

            # Fair price from imbalance
            curr_price = self.get_imbalance_price(self.base, self.quote, 0.005)

            # MACD + ATR-based directional bias
            price_multiplier = (0.5 * macd[-1] + 0.5 * macd_[-1]) / atr_[-1] / 50
            price_multiplier = max(price_multiplier, 0.001)
            price_center = curr_price * (1 + price_multiplier)

            # Position skew
            opened_positions = self.get_opened_positions()
            skew_ratio = np.clip(opened_positions / self.max_position, -1.5, 1.5)
            price_center += half_spread * skew_ratio  # shift center away from inventory

            # Symmetric quotes around skewed center
            bid_price = price_center - half_spread
            ask_price = price_center + half_spread

            # Emergency widening if extreme position
            if abs(opened_positions) >= self.max_position:
                if opened_positions > 0:
                    bid_price = price_center - half_spread * 3  # long -> discourage buying
                else:
                    ask_price = price_center + half_spread * 3  # short -> discourage selling

            self.logger.info(f"Quote Center: {round(price_center, 3)} | Spread: {round(half_spread * 2, 4)} | Position: {opened_positions}")
            return bid_price, ask_price

        except Exception as e:
            self.logger.error(f"Error in get_entry_price: {e} | Line: {e.__traceback__.tb_lineno}")
            return 0.0, 0.0


    def calculate_position_size(self, value_entry: float = 10, price: Optional[float] = None) -> float:
        """Calculate position size based on available quote balance and risk pct."""
        balance = self.get_balance().get('data', {}).get(self.quote, {}).get('total', 0) * 10

        if value_entry > balance:
            self.logger.warning("Insufficient balance to open a new position.")
            return 0.0

        price = price or self.get_price(self.base, self.quote)
        if not price or price <= 0:
            self.logger.warning("Invalid price for sizing.")
            return 0.0
        
        quantity = round(value_entry / price, 6)

        return quantity
    
    def execute_trade(self, bid_pr: float, ask_pr: float, quantity: float,
                    order_type: str = 'limit', force: str = 'GTC') -> bool:
        """
        Place concurrent buy and sell orders using ThreadPoolExecutor.
        Ensures delay between repeated executions using latest_entry_ts.
        """
        self.position.cancel_all_orders(self)
        
        def safe_float(value):
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0

        # Cast bid/ask prices to native float to satisfy Cython typing
        bid_pr = float(bid_pr)
        ask_pr = float(ask_pr)

        ticker = self.get_ticker(self.base, self.quote)
        if not ticker:
            self.logger.warning(f"[get_price] No ticker data for {self.base}/{self.quote}")
            return 0.0

        bid = safe_float(ticker.get('bestBid'))
        ask = safe_float(ticker.get('bestAsk'))
        
        bid_pr = min(bid_pr, bid)
        ask_pr = max(ask_pr, ask)

        futures = {
            self.executor.submit(self.buy, size=quantity, price=bid_pr,
                                order_type=order_type, force=force): "buy",
            self.executor.submit(self.sell, size=quantity, price=ask_pr,
                                order_type=order_type, force=force): "sell"
        }

        results: Dict[str, bool] = {"buy": False, "sell": False}

        for future in as_completed(futures):
            side = futures[future]
            try:
                result = future.result()
                results[side] = bool(result)
                self.logger.info(f"{side.capitalize()} result: {result}")
            except Exception as e:
                self.logger.error(f"{side.capitalize()} failed: {e}")
                results[side] = False

        # Optional: summarize the outcome
        if results["buy"] and results["sell"]:
            self.logger.info("Both buy and sell orders executed successfully.")
        elif results["buy"] or results["sell"]:
            self.logger.warning(f"Partial execution: Buy={results['buy']} Sell={results['sell']}")
        else:
            self.logger.warning("Both buy and sell orders failed.")

        return results

    def close_all_remaining_positions(self):
        """Close all remaining positions forcefully."""
        self.logger.info("Closing all remaining orders and positions forcefully...")
        self.position.cancel_all_orders(self, force_close=True)
        self.position.close_all_positions(self, force_close=True)

    def handle_stop_loss(self):
        """Check stop loss condition."""
        curr_balance = self.get_balance().get('data', {}).get(self.quote, {}).get('total', 0)
        if curr_balance < self.init_balance * (1 - self.stop_loss_threshold):
            self.logger.warning("Stop Loss triggered!")
            self.stop_flag = True
            self.handle_exit()

    def run(self):
        """Run main strategy loop."""
        self.close_all_remaining_positions()
        self.logger.info("Starting Strategy...")
        self.init_balance = self.get_balance().get('data', {}).get(self.quote, {}).get('total', 0)
        try:
            while not self.stop_flag:
                bid_pr, ask_pr = self.get_entry_price()
                self.handle_stop_loss()

                if bid_pr != 0 and ask_pr != 0:
                    price = self.get_price(self.base, self.quote, "fair")
                    size = self.calculate_position_size(self.value_entry, price)
                    if size > 0:
                        self.execute_trade(bid_pr, ask_pr, size, order_type='limit')
                    else:
                        self.logger.warning("Zero position size, skipping trade.")
                else:
                    # self.logger.info("No trade or max position reached.")
                    pass

                time.sleep(self.sleep_time)

        except Exception as e:
            self.logger.exception(f"Run error: {e}")

        finally:
            self.handle_exit()
            self.logger.info("Strategy stopped cleanly.")
