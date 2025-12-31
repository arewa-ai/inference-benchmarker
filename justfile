#!# Image Names #!#
IMAGE_NAME := "arewa/inference-benchmarker"

#!# ECR Configuration #!#
ECR_REGISTRY := "419177720094.dkr.ecr.mx-central-1.amazonaws.com"
REGION := "mx-central-1"

#!# Constructed ECR Paths #!#
ECR_REPO := ECR_REGISTRY + "/" + IMAGE_NAME

#!# Login to ECR #!#
ecr-login:
    aws ecr get-login-password --region {{REGION}} | docker login --username AWS --password-stdin {{ECR_REGISTRY}}

#!# Build, tag, and push image #!#
build-push TAG:
    docker build -t {{IMAGE_NAME}}:{{TAG}} .
    docker tag {{IMAGE_NAME}}:{{TAG}} {{ECR_REPO}}:{{TAG}}
    docker push {{ECR_REPO}}:{{TAG}}
