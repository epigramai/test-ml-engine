# How to use ml engine to predict

https://cloud.google.com/sdk/gcloud/reference/ml-engine/models/create
https://cloud.google.com/sdk/gcloud/reference/ml-engine/versions/create

gcloud ml-engine models create test --enable-logging

curl  -H "Content-Type: application/json" -H "Authorization: Bearer $(gcloud auth print-access-token)" -X POST -d '{"name": "v1", "deployment_uri": "gs://test-model-epigram/incv3_batch/1"}' https://ml.googleapis.com/v1/projects/test-gcloud-ml-deploy/models/test/versions

curl  -H "Content-Type: application/json" -H "Authorization: Bearer $(gcloud auth print-access-token)" -X POST -d '{"name": "v2", "deployment_uri": "gs://test-model-epigram/incv3_batch/1", "machine_type":"mls1-highcpu-4"}' https://ml.googleapis.com/v1/projects/test-gcloud-ml-deploy/models/test/versions

curl  -H "Content-Type: application/json" -H "Authorization: Bearer $(gcloud auth print-access-token)" -X POST -d '{"name": "v3", "deployment_uri": "gs://test-model-epigram/incv3_batch/1", "machine_type":"mls1-highcpu-4", "runtime_version": "1.2avx"}' https://ml.googleapis.com/v1/projects/test-gcloud-ml-deploy/models/test/versions

v1 (standard)       [2.47, 1.93, 1.97, 1.89, 2.12] ~ 2.076
v2 (better machine) [0.88, 0.67, 0.64, 0.70, 0.65] ~ 0.708 3x speedup
v3 (1.2avx)         [TODO] ~ TODO