"""Optional OpenTelemetry tracing.

Metrics tell you "p95 is 300 ms". Traces tell you that this particular 300 ms
was 40 ms of decoding, 250 ms of inference and 10 ms of encoding. The first
finds the problem, the second locates it, and neither substitutes for the
other.

Tracing is off unless OTEL_EXPORTER_OTLP_ENDPOINT is set, which is what
docker-compose.yml does. With it unset, opentelemetry-api hands out
non-recording spans and every call below becomes a no-op, so the image still
runs standalone under a bare `docker run` at essentially zero cost.

Only the API package is imported at module scope; it is small and pure
Python. The SDK, the exporter and the FastAPI instrumentation are imported
inside setup(), so a slim install without them still boots.

Note on versions: the core packages and the instrumentation packages use
different numbering, and they must match. Core 1.X.0 pairs with
instrumentation 0.(X+21)b0 — here 1.27.0 with 0.48b0. Mixing them shows up
either as an ImportError or, far harder to diagnose, as spans that silently
never arrive.
"""

import logging
from contextlib import contextmanager
from typing import Optional

from opentelemetry import trace

from app import config

logger = logging.getLogger("yolo-service")

# This is the *instrumentation scope*, not the service name — it says which
# code produced a span (visible as otel.library.name on each span), whereas
# service.name in setup() says which process it came from. Two different
# things, so they should not share a string.
_tracer = trace.get_tracer(__name__)
_enabled = False


def setup(app=None) -> bool:
    """Install the OTLP exporter. Returns whether tracing was switched on."""
    global _enabled

    endpoint = config.otlp_endpoint()
    if not endpoint:
        logger.info(
            '{"event": "tracing_disabled", '
            '"reason": "OTEL_EXPORTER_OTLP_ENDPOINT not set"}'
        )
        return False

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(
            resource=Resource.create({
                "service.name": config.service_name(),
                "service.version": config.service_version(),
            })
        )
        # insecure=True: the collector sits on the compose network, not the
        # public internet, so there is no TLS to negotiate. Note that the gRPC
        # endpoint is host:port with no http:// prefix — adding one is an
        # immediate error.
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
        )
        trace.set_tracer_provider(provider)

        if app is not None:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            # Prometheus scrapes /metrics every 10s and the container
            # healthcheck hits /health even more often. Those spans carry no
            # information and would bury the requests that matter.
            FastAPIInstrumentor.instrument_app(app, excluded_urls="health,metrics")
    except Exception as exc:  # telemetry failing is minor; the service dying is not
        logger.warning('{"event": "tracing_setup_failed", "error": "%s"}', exc)
        return False

    _enabled = True
    logger.info('{"event": "tracing_enabled", "endpoint": "%s"}', endpoint)
    return True


def enabled() -> bool:
    return _enabled


@contextmanager
def span(name: str, **attributes):
    """Open a child span. A no-op when tracing is disabled.

    Span attributes and metric labels follow opposite cardinality rules:
    image.width=1280 is perfectly fine here, because a span is a discrete
    event rather than an aggregated time series. The same value as a metric
    label would be a disaster.
    """
    with _tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
        yield current


def annotate(**attributes) -> None:
    """Attach attributes to whichever span is currently active."""
    current = trace.get_current_span()
    for key, value in attributes.items():
        if value is not None:
            current.set_attribute(key, value)


def current_trace_id() -> Optional[str]:
    """Hex trace id of the active span, 32 characters.

    Written into the access log so that a single error line leads straight to
    the full trace in Jaeger. That join is called log-trace correlation and is
    one of the markers of a mature observability setup.
    """
    context = trace.get_current_span().get_span_context()
    if not context.is_valid:
        return None
    return format(context.trace_id, "032x")
