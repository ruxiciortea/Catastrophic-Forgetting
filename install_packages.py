import subprocess

packages = [
    'huggingface-hub',
    'transformers',
    'sentencepiece',
    'pytorch-lightning',
    'peft'
]

for package in packages:
    subprocess.call(['pip', 'install', '--upgrade', package])
