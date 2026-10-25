"""Kafka serializers shared by every producer in the service.

kafka-python 3 deprecates plain callables as serializers in favour of
`kafka.serializer.Serializer` subclasses, which also receive record headers.
"""
from __future__ import annotations

import json
from typing import Any

from kafka.serializer import Serializer


class JsonSerializer(Serializer):
    def serialize(self, topic: str, headers: list[tuple[str, bytes]], data: Any) -> bytes:
        return json.dumps(data).encode("utf-8")


class StringSerializer(Serializer):
    def serialize(
        self, topic: str, headers: list[tuple[str, bytes]], data: str | None
    ) -> bytes | None:
        return data.encode("utf-8") if data is not None else None
