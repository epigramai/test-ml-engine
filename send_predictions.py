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

    print(response)

    return response['predictions']


if __name__ == '__main__':
    import cv2
    import time

    img = cv2.imread('./img_71509.jpg')
    print(img.shape)
    img_height, img_width, _ = img.shape
    base_heigth = 150
    img = cv2.resize(img, (int((img_width / img_height) * base_heigth), base_heigth))
    print(img.shape)

    total_time = 0

    for i in range(20):
        t = time.time()
        print('doing req {}'.format(i))

        try:
            preds = predict_json('test-gcloud-ml-deploy', 'test', {'inputs': img.tolist()}, version='v1')
        except Exception as e:
            print(e)

        print(len(preds))
        req_time = time.time() - t
        print('took {} seconds'.format(req_time))
        total_time += req_time
        time.sleep(2)

    print('avg {} seconds'.format(total_time / 10))
