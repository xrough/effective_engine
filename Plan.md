# System Blueprint: Options Market Maker MVP

## 1. Project Objective
Build a Minimum Viable Product (MVP) for an event-driven, sell-side options market-making simulation. The system must use a simplified Domain-Driven Design (DDD) and Event-Driven Architecture (EDA). 

**Key Constraints for this MVP:**
* **Single Data Source:** Read underlying tick data from a CSV or a single WebSocket mock.
* **Instrument:** One underlying asset and a few vanilla European options (Calls/Puts).
* **Scope:** Sell-side quoting and basic Delta hedging only. Ignore Gamma, Vega, Theta, and complex order routing.
* **Concurrency:** Use a simple synchronous Observer pattern or a single-thread `asyncio.Queue` for the event bus. Do not over-engineer with Kafka/RabbitMQ.

---

## 2. Event System (The Core Loop)
All components communicate by emitting and listening to Events. 

**Event Types to Implement:**
1. `MarketDataEvent`: Contains `timestamp`, `underlying_price`.
2. `QuoteGeneratedEvent`: Contains `instrument_id`, `bid_price`, `ask_price`, `timestamp`.
3. `TradeExecutedEvent`: Contains `instrument_id`, `side` (Buy/Sell), `price`, `quantity`, `timestamp`.
4. `OrderSubmittedEvent`: Contains `instrument_id`, `side`, `quantity`, `order_type` (Market/Limit).

---

## 3. Core Domain (Pure Logic, No I/O)
Implement these as strict data classes (e.g., Python `@dataclass` or Pydantic) and pure functions.

* **`Instrument`**: Base class. Subclasses: `Underlying` and `Option` (strike, expiry, call/put flag).
* **`PositionManager`**: Tracks current inventory (e.g., `{'AAPL': -100, 'AAPL_150_C': 10}`). Updates state when a `TradeExecutedEvent` occurs.
* **`PricingEngine`**: Implement a basic Black-Scholes formula to calculate the theoretical price ("theo") and Delta for an option given the current underlying price, time to expiry, fixed volatility, and risk-free rate.

---

## 4. Adapters & Simulators
* **`MarketDataAdapter`**: Reads a mock CSV of underlying prices row-by-row and publishes `MarketDataEvent`.
* **`ProbabilisticTaker` (Market Simulator)**: Listens to `QuoteGeneratedEvent`. Uses a simple probability function (e.g., 10% chance to trade) to randomly "hit" the bid or "lift" the ask. If a trade happens, it publishes a `TradeExecutedEvent`.

---

## 5. Strategy & Workflow (Business Logic)
* **`QuoteEngine` (Sell-Side Strategy)**:
    * Listens to `MarketDataEvent`.
    * Calls `PricingEngine` to calculate the option's Theo.
    * Applies a fixed spread (e.g., Theo - 0.05 for Bid, Theo + 0.05 for Ask).
    * Publishes `QuoteGeneratedEvent`.
* **`DeltaHedger` (Risk Workflow)**:
    * Listens to `TradeExecutedEvent`.
    * Calculates the aggregate portfolio Delta: $\Delta_{portfolio} = \Delta_{underlying} \times Position_{underlying} + \sum (\Delta_{option} \times Position_{option})$
    * If $|\Delta_{portfolio}|$ exceeds a predefined threshold (e.g., 50 delta), it publishes an `OrderSubmittedEvent` (Market order on the underlying) to flatten the directional risk.

---

## 6. Implementation Instructions for Coding AI
1. **Scaffold the Event Bus first:** Create a simple pub/sub dispatcher.
2. **Implement Core Domain models:** Ensure strict typing.
3. **Build the Black-Scholes `PricingEngine`.**
4. **Implement the Strategies and Adapters.**
5. **Create a `main.py` entry point:** Wire the publishers and subscribers together, load a dummy array of market data, and run the simulation loop, printing logs to the console to prove the flow works.