from googleapiclient import discovery


def predict_json(project, model, instances, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    # Create the ML Engine service object.
    # To authenticate set the environment variable
    # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
    service = discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']


if __name__ == '__main__':
    import cv2
    import time

    PROJECT = 'test-gcloud-ml-deploy'
    MODEL = 'test'
    VERSION = 'v1'
    NUM_REQUESTS = 10

    img = cv2.imread('./img_71509.jpg')
    print(img.shape)

    # Uncomment this if you want to resize image
    # img_height, img_width, _ = img.shape
    # base_heigth = 50
    # img = cv2.resize(img, (int((img_width / img_height) * base_heigth), base_heigth))
    # print(img.shape)

    total_time = 0

    for i in range(NUM_REQUESTS):
        t = time.time()
        print('doing req {}'.format(i))

        try:
            preds = predict_json(PROJECT, MODEL, {'inputs': img.tolist()}, version=VERSION)
            req_time = time.time() - t
            print(len(preds))
            print('took {} seconds'.format(req_time))
        except Exception as e:
            req_time = time.time() - t
            print(e)

        total_time += req_time
        time.sleep(2)

    print('avg {} seconds'.format(total_time / NUM_REQUESTS))
