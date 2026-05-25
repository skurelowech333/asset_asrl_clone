from setuptools import setup, find_packages
from asset_asrl._build_extensions import build_extensions

setup(
    name="asset_asrl",
    version="0.1.0",

    packages=find_packages(include=["asset_asrl", "asset_asrl.*"]),

    ext_modules=build_extensions(),

    include_package_data=True,
    zip_safe=False,
)
