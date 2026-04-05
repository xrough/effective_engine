# How the handlers are registered and how events are published accordingly?

## Step 1 — Registration (register_handlers)

For example, in VarianceAlphaSignal:


void register_handlers() {
    bus_->subscribe<events::OptionMidQuoteEvent>(
        [this](const events::OptionMidQuoteEvent& e) { on_option_quote(e); }
    );
}
This calls subscribe<T>() on the bus with a lambda. Inside subscribe:


template<typename T>
void subscribe(std::function<void(const T&)> handler) {
    auto key = std::type_index(typeid(T));   // key = OptionMidQuoteEvent's type
    handlers_[key].emplace_back(
        [h = std::move(handler)](const std::any& event) {
            h(std::any_cast<const T&>(event));  // T baked in here
        }
    );
}
Two things happen:

typeid(OptionMidQuoteEvent) produces a unique type key
The lambda [this](...){ on_option_quote(e); } gets wrapped in another lambda that knows how to cast any → OptionMidQuoteEvent, then stored in the map
After this call, handlers_ looks like:


{
  type_index(OptionMidQuoteEvent) → [ wrapper_lambda_1 ]
}


## Step 2 — The map holds it permanently

handlers_ is a member of EventBus (EventBus.hpp:98-101):


unordered_map<
    type_index,
    vector<function<void(const std::any&)>>
> handlers_;
The registration lives here for the lifetime of the bus. Multiple calls to subscribe for the same type append to the vector — that's how multiple components can all listen to the same event.

## Step 3 — Delivery (publish)

When ImpliedVarianceExtractor updates and VarianceAlphaSignal (or whoever) calls publish (EventBus.hpp:72-84):


template<typename T>
void publish(const T& event) {
    auto key = std::type_index(typeid(T));   // same key as subscribe
    auto it = handlers_.find(key);           // look up the vector
    std::any wrapped(event);                 // one copy into any
    for (auto& handler_wrapper : it->second) {
        handler_wrapper(wrapped);            // calls each lambda in order
    }
}
Each handler_wrapper unwraps any → OptionMidQuoteEvent and calls on_option_quote(e) synchronously.

## Full picture

register_handlers() called at startup
    subscribe<OptionMidQuoteEvent>(lambda)
        → key  = type_index(OptionMidQuoteEvent)
        → wrap = any→cast→lambda
        → handlers_[key].push_back(wrap)       ← stored here forever

... later, new quote arrives ...

publish<OptionMidQuoteEvent>(e)
    → key = type_index(OptionMidQuoteEvent)    ← same key
    → handlers_[key] found
    → wrap(any(e))
        → any_cast<OptionMidQuoteEvent>
        → on_option_quote(e)                   ← your code runs
The type is the key — no string names, no IDs. If you never call register_handlers(), the lambda is never stored, and publish silently does nothing (the handlers_.find(key) returns end()).