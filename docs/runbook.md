# Echo Ridge Scoring Engine - Operations Runbook

## Overview

This runbook provides operational guidance for the Echo Ridge AI-readiness scoring service, including deployment procedures, monitoring setup, troubleshooting, and incident response.

**Service**: Echo Ridge Scoring Engine  
**Version**: 1.0.0  
**Last Updated**: 2024-01-15  
**On-Call Contact**: Echo Ridge Team

## Service Level Objectives (SLOs)

### Performance SLOs

| Metric | Target | Measurement Window |
|--------|--------|-------------------|
| **API Response Time (p99)** | < 150ms | 5-minute rolling window |
| **API Response Time (p95)** | < 100ms | 5-minute rolling window |
| **Batch Processing Throughput** | > 1000 companies/minute | 10-minute window |
| **Service Availability** | 99.9% uptime | Monthly |

### Quality SLOs

| Metric | Target | Measurement Window |
|--------|--------|-------------------|
| **Error Rate** | < 0.1% (1 in 1000 requests) | 5-minute rolling window |
| **Scoring Consistency** | Same input = same output | 100% deterministic |
| **Data Freshness** | Normalization context < 24h old | Continuous |

### Resource SLOs

| Metric | Target | Measurement Window |
|--------|--------|-------------------|
| **Memory Usage** | < 2GB per instance | Continuous |
| **CPU Utilization** | < 70% average | 5-minute window |
| **Database Response Time** | < 50ms p95 | 5-minute window |

## Error Budgets

**Monthly Error Budget**: 0.1% of requests (approximately 43 minutes of downtime per month)

**Error Budget Burn Rate Alerts**:
- **Fast Burn**: > 10x normal rate (triggers immediate alert)  
- **Slow Burn**: > 2x normal rate (triggers daily summary)

## Hybrid Scoring Operations

### Roman Integration Monitoring

When integrated with Roman's agentic pipeline, monitor these additional metrics:

**Adapter Performance**:
- PlaceNorm → CompanySchema conversion time (target: < 10ms)
- Warning rate per conversion (expect 60-80% due to missing quantitative data)
- Failed conversions requiring manual intervention

**Blending Quality**:
- AI vs Deterministic score divergence rates by strategy
- Divergence flag frequency (> 30% may indicate model drift)
- Blending confidence distribution

**Integration Health**:
```bash
# Check adapter warnings
grep "WARNING.*adapter" /var/log/echo_ridge.log | tail -20

# Monitor divergence patterns  
grep "divergence.*threshold" /var/log/echo_ridge.log | wc -l

# Validate deterministic consistency
echo_ridge_cli validate --input sample_roman_data.jsonl
```

**Alert Thresholds**:
- Adapter failure rate > 5%: CRITICAL
- Divergence flags > 50%: WARNING  
- Blending time > 50ms: WARNING

### Hybrid Score Interpretation

**Divergence Analysis**:
- `< 0.2`: Normal variation, no action needed
- `0.2 - 0.4`: Monitor trend, may indicate data quality issues  
- `> 0.4`: Significant divergence, investigate model assumptions

**Warning Categories**:
- Missing quantitative data (expected for Roman integration)
- Low confidence extraction (< 0.6) 
- Estimation fallbacks used

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │────│   FastAPI App   │────│   PostgreSQL    │
│   (nginx/ALB)   │    │   (uvicorn)     │    │   Database      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              │
                       ┌─────────────────┐
                       │  Scoring Engine │
                       │  (src/scoring)  │
                       └─────────────────┘
```

### Key Components

1. **FastAPI Application** (`src/api/`) - REST API endpoints
2. **Scoring Engine** (`src/scoring.py`) - Core D/O/I/M/B scoring logic  
3. **Batch Processor** (`src/batch.py`) - JSONL batch processing
4. **Persistence Layer** (`src/persistence.py`) - Database interactions
5. **Monitoring** (`src/monitoring.py`) - Structured logging and metrics

## Deployment Procedures

### Pre-Deployment Checklist

- [ ] All tests pass (`poetry run pytest`)
- [ ] Code linting clean (`poetry run ruff check src/`)
- [ ] Type checking clean (`poetry run mypy src/`)
- [ ] Database migrations applied (if any)
- [ ] Weights configuration frozen (`weights.yaml` version locked)
- [ ] Performance regression testing completed
- [ ] Dependency security scan clean

### Deployment Steps

1. **Staging Deployment**
   ```bash
   # Deploy to staging environment
   kubectl apply -f k8s/staging/
   
   # Verify health
   curl https://scoring-staging.echo-ridge.com/healthz
   
   # Run integration tests
   poetry run pytest tests/integration/ --env=staging
   ```

2. **Production Deployment** (Blue/Green)
   ```bash
   # Deploy new version (green)
   kubectl apply -f k8s/production/green/
   
   # Verify health on green
   curl https://scoring-green.echo-ridge.com/healthz
   
   # Switch traffic gradually
   kubectl patch service scoring-service --patch '{"spec":{"selector":{"version":"green"}}}'
   
   # Monitor metrics for 10 minutes
   # If stable, tear down blue environment
   ```

3. **Rollback Procedure**
   ```bash
   # Immediate rollback to previous version
   kubectl patch service scoring-service --patch '{"spec":{"selector":{"version":"blue"}}}'
   
   # Verify service health
   curl https://scoring.echo-ridge.com/healthz
   ```

### Configuration Management

**Environment Variables**:
```bash
# Required
DATABASE_URL=postgresql://user:pass@host:5432/echo_ridge
WEIGHTS_CONFIG_PATH=/app/weights.yaml

# Optional
LOG_LEVEL=INFO
PERFORMANCE_MONITORING_ENABLED=true
DRIFT_DETECTION_ENABLED=true
```

## Monitoring and Alerting

### Key Metrics to Monitor

1. **Request Metrics**
   - Request rate (requests/second)
   - Response time (p50, p95, p99)
   - Error rate by endpoint
   - Request payload sizes

2. **Scoring Metrics** 
   - Scoring operation latency
   - Batch processing throughput
   - Normalization context age
   - Score distribution stats

3. **System Metrics**
   - CPU and memory usage
   - Database connection pool stats
   - Disk usage and I/O
   - Network latency

4. **Business Metrics**
   - Companies scored per hour
   - API key usage by customer
   - Feature usage patterns
   - Drift detection alerts

### Alert Configuration

#### Critical Alerts (Page Immediately)

```yaml
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.001
  for: 2m
  summary: "Error rate above SLO ({{$value}}%)"

- alert: HighLatency  
  expr: histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) > 0.15
  for: 3m
  summary: "p99 latency above SLO ({{$value}}s)"

- alert: ServiceDown
  expr: up{job="echo-ridge-scoring"} == 0
  for: 1m
  summary: "Service instance down"
```

#### Warning Alerts (Investigate within 4 hours)

```yaml
- alert: DriftDetected
  expr: drift_alerts_total > 0
  for: 5m
  summary: "Model drift detected"

- alert: HighMemoryUsage
  expr: process_resident_memory_bytes > 2e9
  for: 10m 
  summary: "Memory usage above threshold"

- alert: DatabaseSlowQueries
  expr: rate(database_query_duration_seconds{quantile="0.95"}[5m]) > 0.05
  for: 5m
  summary: "Database queries running slowly"
```

### Log Analysis

**Structured Log Format**:
```json
{
  "timestamp": "2024-01-15T10:30:00.123Z",
  "level": "info",
  "service": "echo-ridge-scoring",
  "version": "1.0.0", 
  "component": "api",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
  "event_type": "request_end",
  "message": "POST /score completed",
  "endpoint": "/score",
  "method": "POST",
  "status_code": 200,
  "duration_ms": 45.2,
  "company_id": "acme-corp-001"
}
```

**Common Log Queries**:
```bash
# Find slow requests (>100ms)
grep '"duration_ms":[1-9][0-9][0-9]' application.log

# Error analysis
grep '"level":"error"' application.log | jq '.error_type' | sort | uniq -c

# Request volume by endpoint
grep '"event_type":"request_end"' application.log | jq '.endpoint' | sort | uniq -c
```

## Troubleshooting Guide

### Common Issues

#### 1. High Response Times

**Symptoms**: API p99 latency > 150ms

**Investigation Steps**:
1. Check current request volume: `grep "request_start" logs/ | tail -100`
2. Identify slow operations: `grep "duration_ms.*[5-9][0-9][0-9]" logs/`
3. Check database performance: Query `pg_stat_activity` for long-running queries
4. Verify normalization context age: `GET /stats` endpoint

**Resolution**:
- Scale horizontally if CPU-bound
- Optimize database queries if DB-bound  
- Update normalization context if stale
- Consider caching for repeated requests

#### 2. Scoring Inconsistency

**Symptoms**: Same input producing different scores

**Investigation Steps**:
1. Check normalization context version: `GET /stats`
2. Verify weights configuration hasn't changed: `cat weights.yaml`
3. Check for recent drift alerts: `grep "drift_alert" logs/`
4. Run deterministic validation: `poetry run python cli.py validate-deterministic`

**Resolution**:
- Reload consistent normalization context
- Revert weights to known good version
- Investigate data pipeline changes
- Update drift detection thresholds if false positive

#### 3. Database Connection Issues

**Symptoms**: "Connection refused" or timeout errors

**Investigation Steps**:
1. Check database health: `pg_isready -h $DB_HOST -p $DB_PORT`
2. Verify connection pool: Check `DATABASE_MAX_CONNECTIONS` setting
3. Check for connection leaks: Monitor open connections over time
4. Review database logs for errors

**Resolution**:
- Restart application to reset connection pool
- Scale database if resource-constrained
- Fix connection leaks in application code
- Adjust connection pool settings

#### 4. Memory Leaks

**Symptoms**: Memory usage increasing over time

**Investigation Steps**:
1. Monitor memory metrics: `GET /stats` endpoint
2. Check for large batch operations: `grep "batch.*companies.*[5-9][0-9][0-9]" logs/`
3. Profile memory usage: Use Python memory profilers
4. Check for unclosed resources: Database connections, file handles

**Resolution**:
- Restart application as temporary fix
- Implement streaming for large batches
- Add explicit resource cleanup
- Set memory limits to prevent OOM

### Escalation Procedures

#### Severity 1 (Service Down)
1. **Immediate**: Page on-call engineer
2. **Within 5 minutes**: Assess impact and begin mitigation
3. **Within 15 minutes**: Engage engineering manager if not resolved
4. **Within 30 minutes**: Consider emergency rollback

#### Severity 2 (Degraded Performance)  
1. **Within 15 minutes**: Investigate root cause
2. **Within 1 hour**: Implement mitigation or escalate to Severity 1
3. **Within 4 hours**: Resolve or schedule maintenance window

#### Severity 3 (Feature Issues)
1. **Within 4 hours**: Investigate during business hours
2. **Within 1 business day**: Resolve or create fix plan
3. **Next release cycle**: Deploy permanent fix

## Maintenance Procedures

### Regular Maintenance Tasks

#### Daily
- [ ] Check error rate dashboard
- [ ] Verify backup completion
- [ ] Review security scan results
- [ ] Monitor drift detection alerts

#### Weekly  
- [ ] Review performance trends
- [ ] Update dependency security scan
- [ ] Check disk usage and log rotation
- [ ] Validate normalization context freshness

#### Monthly
- [ ] Review and update weights configuration
- [ ] Calibration analysis with new labeled data
- [ ] Performance capacity planning review
- [ ] Security patches and updates

### Database Maintenance

**Backup Strategy**:
- Automated daily backups with 30-day retention
- Weekly full backups with 1-year retention
- Point-in-time recovery capability

**Index Maintenance**:
```sql
-- Check for unused indexes
SELECT schemaname, tablename, attname, n_distinct, correlation 
FROM pg_stats WHERE schemaname = 'public';

-- Rebuild large indexes monthly
REINDEX INDEX CONCURRENTLY scoring_results_company_id_idx;
```

### Weight Configuration Updates

1. **Backup Current Configuration**
   ```bash
   cp weights.yaml weights-backup-$(date +%Y%m%d).yaml
   ```

2. **Update with Calibration Evidence**
   ```bash
   # Run calibration analysis
   poetry run python -m src.calibration --input labeled_data.jsonl --output calibration_results.json
   
   # Update weights.yaml based on results
   # Increment version number
   # Add calibration evidence
   ```

3. **Validate New Configuration**
   ```bash
   # Test with sample data
   poetry run pytest tests/test_calibration.py
   
   # Deploy to staging first
   # Monitor performance metrics
   # Deploy to production if validated
   ```

## Performance Optimization

### API Optimization

1. **Response Caching**: Implement Redis cache for repeated requests
2. **Request Batching**: Optimize single-request latency over batch throughput
3. **Database Connection Pooling**: Tune pool size based on load testing
4. **Async Processing**: Use async/await for I/O bound operations

### Scoring Pipeline Optimization

1. **Vectorization**: Use NumPy operations where possible
2. **Memory Management**: Explicit cleanup of large objects
3. **Normalization Caching**: Cache normalization parameters
4. **Schema Validation**: Optimize Pydantic validation for hot paths

### Monitoring Overhead

1. **Sampling**: Use sampling for high-volume trace collection
2. **Async Logging**: Non-blocking structured logging
3. **Metric Aggregation**: Pre-aggregate metrics at application level
4. **Log Rotation**: Automated cleanup of old log files

## Security Considerations

### API Security

- **Authentication**: API key validation for all endpoints
- **Rate Limiting**: Per-customer rate limits to prevent abuse  
- **Input Validation**: Strict schema validation on all inputs
- **Output Sanitization**: No sensitive data in error messages

### Data Security

- **Encryption**: Data encrypted at rest and in transit
- **Access Control**: Role-based database access
- **Audit Logging**: All data access logged
- **Data Retention**: Automatic cleanup of old scoring data

### Infrastructure Security

- **Network Segmentation**: Database in private subnet
- **Security Groups**: Minimal required port access
- **Regular Updates**: OS and dependency security patches
- **Vulnerability Scanning**: Automated security scans

## Contact Information

**Primary On-Call**: Echo Ridge Engineering Team  
**Escalation**: Engineering Manager  
**Business Contact**: Product Team  

**Emergency Procedures**: Follow company incident response playbook

**Documentation Updates**: Update this runbook after major changes or incidents

---

*This runbook is a living document. Please update it as the system evolves and new operational knowledge is gained.*