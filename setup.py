from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="RAG_Medical_Chatbot",
    version="0.1",
    author="Krushil",
    packages=find_packages(),
    install_requires = requirements
)

