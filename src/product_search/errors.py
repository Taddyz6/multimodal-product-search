"""Stable service errors and input validation; accepted text is not rewritten."""


class ServiceError(Exception):
    code = "internal_error"
    http_status = 500

    def __init__(self, message=None):
        super().__init__(message or self.code)


class InvalidInput(ServiceError):
    code = "invalid_input"
    http_status = 422


class Unavailable(ServiceError):
    code = "online_unavailable"
    http_status = 503


class EncodeFailure(ServiceError):
    code = "query_encoding_failed"
    http_status = 502


class Busy(ServiceError):
    code = "service_busy"
    http_status = 429


class QueryProcessor:
    """Validate without rewriting, stripping or normalising accepted text."""

    def validate(self, query, top_k):
        if (
            not isinstance(query, str)
            or not query.strip()
            or len(query) > 2048
            or any((ord(c) < 32 and c not in "\t\n\r" for c in query))
        ):
            raise InvalidInput(
                "Enter a query of up to 2,048 characters without binary control characters."
            )
        if type(top_k) is not int or not 1 <= top_k <= 100:
            raise InvalidInput("top_k must be an integer from 1 to 100.")
        return query
