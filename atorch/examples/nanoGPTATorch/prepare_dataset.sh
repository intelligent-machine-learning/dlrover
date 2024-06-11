
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type dataset --resume-download openwebtext --local-dir openwebtext

cd openwebtext/subsets/
for tarfile in *.tar; do
    tar -xvf "$tarfile"
done
cd ../../
pip install datasets tiktoken
python openwebtext/prepare.py