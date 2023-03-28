FROM registry.cn-hangzhou.aliyuncs.com/dlrover_deeprec/deeprec:v11
COPY dist/dlrover-0.1.0-py3-none-any.whl /home/dlrover-0.1.0-py3-none-any.whl
RUN pip install /home/dlrover-0.1.0-py3-none-any.whl -I --no-deps
RUN pip install pyhocon 
COPY model_zoo /home/model_zoo
