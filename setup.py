from setuptools import find_packages, setup

setup(
    name="thermofisher-coa-agent",
    version="1.0.0",
    description="Enterprise AI Chat Agent for ThermoFisher / Fisher Scientific Certificates of Analysis (CoA) Search & Quality Verification",
    packages=find_packages(include=["thermofisher_agent", "thermofisher_agent.*"]),
    python_requires=">=3.9",
    install_requires=[
        "pydantic>=2.7.0",
        "fastapi>=0.111.0",
        "uvicorn[standard]>=0.30.0",
    ],
    entry_points={
        "console_scripts": [
            "thermofisher-coa-agent=thermofisher_agent.cli:main",
        ],
    },
)
