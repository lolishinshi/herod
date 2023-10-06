import os
import tomlkit
from pydantic import BaseModel
from pydantic_settings import BaseSettings


def xdg_data_home():
    return os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))


def xdg_config_home():
    return os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))


class MilvusConfig(BaseModel):
    host: str = "localhost"
    port: int = 19530


class Config(BaseSettings):
    milvus: MilvusConfig = MilvusConfig()


def load_config():
    config_file = os.path.join(xdg_config_home(), "herod", "config.toml")
    if os.path.exists(config_file):
        with open(config_file) as f:
            config = tomlkit.parse(f.read())
        return Config(**config)
    else:
        return Config()


config = load_config()
