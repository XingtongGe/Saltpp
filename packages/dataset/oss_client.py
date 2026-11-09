import os
from urllib.parse import urlparse

import alibabacloud_oss_v2 as oss
import yaml


DEFAULT_OSS_CONFIG_PATH = os.path.expanduser("~/.credentials/oss_config.yaml")
DEFAULT_OSS_REGION = os.environ.get("OSS_REGION", "us-east-1")
DEFAULT_OSS_ENDPOINT = os.environ.get("OSS_ENDPOINT", "")


def _load_oss_config(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"OSS config must be a mapping: {path}")
    return data


class Client:
    """Object storage client wrapper for dataset loading."""

    def __init__(
        self,
        conf_path=None,
        region=None,
        endpoint=None,
        connect_timeout=None,
        readwrite_timeout=None,
        max_attempts=None,
        max_connections=None,
    ):
        self.conf_path = conf_path if conf_path and os.path.exists(conf_path) else DEFAULT_OSS_CONFIG_PATH
        self.config = _load_oss_config(self.conf_path)
        self.connect_timeout = connect_timeout
        self.readwrite_timeout = readwrite_timeout
        self.max_attempts = max_attempts
        self.max_connections = max_connections
        self.region = (
            region
            or os.environ.get("OSS_REGION")
            or os.environ.get("ALIBABA_CLOUD_REGION_ID")
            or self.config.get("region")
            or DEFAULT_OSS_REGION
        )
        self.endpoint = endpoint or os.environ.get("OSS_ENDPOINT") or self.config.get("endpoint") or DEFAULT_OSS_ENDPOINT
        self.access_key_id = os.environ.get("OSS_ACCESS_KEY_ID") or self.config.get("access_key_id")
        self.access_key_secret = os.environ.get("OSS_ACCESS_KEY_SECRET") or self.config.get("access_key_secret")
        self.security_token = os.environ.get("OSS_SECURITY_TOKEN") or self.config.get("security_token")
        self.bucket_aliases = self.config.get("bucket_aliases") or {}
        self._client = None

    def _get_local_client(self):
        return self

    def _oss_client(self):
        if self._client is not None:
            return self._client
        if not self.region:
            raise RuntimeError(
                "OSS region is required. Set OSS_REGION and optionally OSS_ENDPOINT."
            )
        cfg = oss.config.load_default()
        if self.access_key_id and self.access_key_secret:
            cfg.credentials_provider = oss.credentials.StaticCredentialsProvider(
                self.access_key_id,
                self.access_key_secret,
                self.security_token,
            )
        else:
            cfg.credentials_provider = oss.credentials.EnvironmentVariableCredentialsProvider()
        cfg.region = self.region
        if self.endpoint:
            cfg.endpoint = self.endpoint
        if self.connect_timeout is not None:
            cfg.connect_timeout = self.connect_timeout
        if self.readwrite_timeout is not None:
            cfg.readwrite_timeout = self.readwrite_timeout
        if self.max_attempts is not None:
            cfg.retry_max_attempts = self.max_attempts
        if self.max_connections is not None:
            http_kwargs = {"max_connections": self.max_connections}
            if self.connect_timeout is not None:
                http_kwargs["connect_timeout"] = self.connect_timeout
            if self.readwrite_timeout is not None:
                http_kwargs["readwrite_timeout"] = self.readwrite_timeout
            cfg.http_client = oss.transport.RequestsHttpClient(**http_kwargs)
        self._client = oss.Client(cfg)
        return self._client

    def _parse_uri(self, uri):
        if not isinstance(uri, str):
            raise TypeError(f"Storage uri must be str, got {type(uri)!r}")
        parsed = urlparse(uri)
        if parsed.scheme in ("s3", "oss"):
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
        elif parsed.scheme in ("http", "https"):
            parts = parsed.netloc.split(".")
            bucket = parts[0]
            key = parsed.path.lstrip("/")
        else:
            raise ValueError(
                f"Unsupported storage uri format. Expected s3://bucket/key or oss://bucket/key; got {uri!r}"
            )
        if not bucket or not key:
            raise ValueError(f"Storage uri must include bucket and key: {uri!r}")
        bucket = self.bucket_aliases.get(bucket, bucket)
        return bucket, key

    def get(self, uri):
        if isinstance(uri, str) and os.path.isfile(uri):
            with open(uri, "rb") as f:
                return f.read()
        bucket, key = self._parse_uri(uri)
        result = self._oss_client().get_object(oss.GetObjectRequest(bucket=bucket, key=key))
        with result.body as body_stream:
            return body_stream.read()

    def Get(self, uri):
        return self.get(uri)

    def put(self, uri, data):
        bucket, key = self._parse_uri(uri)
        return self._oss_client().put_object(
            oss.PutObjectRequest(bucket=bucket, key=key, body=data)
        )

    def Put(self, uri, data):
        return self.put(uri, data)
