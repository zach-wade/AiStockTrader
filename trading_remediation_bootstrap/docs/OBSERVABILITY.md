# Observability Baseline

## Logs
- Structured JSON logs with fields: ts, level, module, event, order_id, symbol, qty, price, error.

## Metrics (examples)
- data_freshness_seconds{source}
- job_success_total{job}
- order_submit_success_total{broker}
- order_latency_ms{broker}
- risk_check_fail_total{reason}
- exception_total{module}

## Alerts
- Data freshness > 120s
- Error rate > 1% over 5m
- Order submit success < 95% over 15m
- Drawdown > X%
