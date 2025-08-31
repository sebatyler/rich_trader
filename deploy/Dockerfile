FROM ghcr.io/atflow-corp/docker/python_docker:3.11

# skip checking python version
ENV ZAPPA_RUNNING_IN_DOCKER=True

ENV PYTHONUNBUFFERED=1
ENV PIP_TIMEOUT=3
ENV DJANGO_ALLOW_ASYNC_UNSAFE=true

# Define function directory
ARG FUNCTION_DIR="/function"

# Create function directory
RUN mkdir -p ${FUNCTION_DIR}

# Set working directory to function root directory
WORKDIR ${FUNCTION_DIR}

RUN python -m venv venv_deploy
ENV PATH=${FUNCTION_DIR}/venv_deploy/bin:$PATH

ARG USE_CACHE=1
ARG PACKAGE_FILE=requirements.txt
ARG TARGET=prod

COPY ${PACKAGE_FILE} ./

# Install packages with cache
RUN --mount=type=cache,target=/venv_cache,id=${PACKAGE_FILE}.rich_trader.deploy \
    cp -a /venv_cache/* ${FUNCTION_DIR}/venv_deploy/; \
    pip install uv && uv pip install --no-cache-dir -r ${PACKAGE_FILE} && zappa -v; \
    [ $USE_CACHE -eq 1 ] && cp -a ${FUNCTION_DIR}/venv_deploy/* /venv_cache/ || echo readonly cache

# AWS confidential from arguments to run 'aws s3'
ENV AWS_DEFAULT_REGION=ap-northeast-2
RUN --mount=type=secret,id=aws_access_key_id \
    --mount=type=secret,id=aws_secret_access_key \
    aws configure --profile sebatyler set aws_access_key_id $(cat /run/secrets/aws_access_key_id) && \
    aws configure --profile sebatyler set aws_secret_access_key $(cat /run/secrets/aws_secret_access_key) && \
    aws configure --profile sebatyler set region ${AWS_DEFAULT_REGION}

COPY ./ ./

# download files
RUN bash download.sh && mv -v .env.${TARGET} .env

# Collect static files
RUN AWS_PROFILE=sebatyler DJANGO_SETTINGS_MODULE=rich_trader.settings.${TARGET} USE_DB_URL=0 \
    python manage.py collectstatic --noinput

# Grab the zappa handler.py and put it in the working directory
RUN ZAPPA_HANDLER_PATH=$(python -c "from zappa import handler; print (handler.__file__)") \
    && echo $ZAPPA_HANDLER_PATH \
    && cp $ZAPPA_HANDLER_PATH ./

RUN cp -a ${FUNCTION_DIR}/deploy/aws-lambda-rie /usr/local/bin/; cp -a ${FUNCTION_DIR}/deploy/entrypoint.sh /

ENTRYPOINT [ "/entrypoint.sh" ]

CMD ["handler.lambda_handler"]
