import numpy as np


def batch_predict(model, X, predict_batch_size=4096, output_dim=1):
    """
    Apply batch predicting to speed up the inference.
    :param model: model: tensorflow.python.keras.engine.training.Model
    :param X: original input features X: list
    :param predict_batch_size: batch size for predicting: int
    :return: predict results: list
    """
    predict_size = len(X[0])
    preds = np.zeros(shape=(predict_size, output_dim))
    predict_range = range(0, predict_size, predict_batch_size)
    for i in predict_range[:-1]:
        predict_batch_data = [arr[i:i + predict_batch_size] for arr in X]
        preds[i:i + predict_batch_size] = model.predict_on_batch(predict_batch_data)
    predict_batch_data = [arr[predict_range[-1]:predict_size] for arr in X]
    preds[predict_range[-1]:predict_size] = model.predict_on_batch(predict_batch_data)

    return preds


def mt_batch_predict(model, X, predict_batch_size=4096, output_dim=1):
    """
    Apply multi-task batch predicting to speed up the inference.
    :param model: model: tensorflow.python.keras.engine.training.Model
    :param X: original input features X: list
    :param predict_batch_size: batch size for predicting: int
    :return: predict results including predicted CTR and predicted market price distribution: (list, list)
    """
    predict_size = len(X[0])
    ctr_preds = np.zeros(shape=(predict_size, 1))
    mp_preds = np.zeros(shape=(predict_size, output_dim))
    predict_range = range(0, predict_size, predict_batch_size)
    for i in predict_range[:-1]:
        predict_batch_data = [arr[i:i + predict_batch_size] for arr in X]
        pred_result = model.predict_on_batch(predict_batch_data)
        ctr_preds[i:i + predict_batch_size] = pred_result[0]
        mp_preds[i:i + predict_batch_size] = pred_result[1]
    predict_batch_data = [arr[predict_range[-1]:predict_size] for arr in X]
    pred_result = model.predict_on_batch(predict_batch_data)
    ctr_preds[predict_range[-1]:predict_size] = pred_result[0]
    mp_preds[predict_range[-1]:predict_size] = pred_result[1]

    return ctr_preds, mp_preds


def light_mp_batch_predict(model, X, Z, predict_batch_size=4096, output_dim=301):
    """
    Apply batch predicting to speed up the inference.
    :param model: model: tensorflow.python.keras.engine.training.Model
    :param X: original input features X: list
    :param predict_batch_size: batch size for predicting: int
    :return: predict results: list
    """
    predict_size = len(X[0])
    Z_pred = np.zeros(shape=(predict_size,))
    Z_prob = np.zeros(shape=(predict_size,))
    predict_range = range(0, predict_size, predict_batch_size)
    weights = np.array([i for i in range(output_dim)])
    pdf_sum = np.array([0] * output_dim, dtype=np.float64)
    for i in predict_range[:-1]:
        predict_batch_data = [arr[i:i + predict_batch_size] for arr in X]
        batch_z_index = Z[i:i + predict_batch_size]
        row_index = np.arange(len(batch_z_index))
        pred_pdf = model.predict_on_batch(predict_batch_data)
        Z_prob[i:i + predict_batch_size] = pred_pdf[row_index, batch_z_index]
        Z_pred[i:i + predict_batch_size] = pred_pdf.dot(weights)
        pdf_sum += pred_pdf.sum(axis=0)
    predict_batch_data = [arr[predict_range[-1]:predict_size] for arr in X]
    batch_z_index = Z[predict_range[-1]:predict_size]
    row_index = np.arange(len(batch_z_index))
    pred_pdf = model.predict_on_batch(predict_batch_data)
    Z_prob[predict_range[-1]:predict_size] = pred_pdf[row_index, batch_z_index]
    Z_pred[predict_range[-1]:predict_size] = pred_pdf.dot(weights)
    pdf_sum += pred_pdf.sum(axis=0)
    pdf_avg = pdf_sum/predict_size
    return Z_pred, Z_prob, pdf_avg


def light_mt_batch_predict(model, X, Z, predict_batch_size=4096, output_dim=301):
    """
    Apply multi-task batch predicting to speed up the inference.
    :param model: model: tensorflow.python.keras.engine.training.Model
    :param X: original input features X: list
    :param predict_batch_size: batch size for predicting: int
    :return: predict results including predicted CTR and predicted market price distribution: (list, list)
    """
    predict_size = len(X[0])
    ctr_preds = np.zeros(shape=(predict_size, 1))
    Z_pred = np.zeros(shape=(predict_size,))
    Z_prob = np.zeros(shape=(predict_size,))
    predict_range = range(0, predict_size, predict_batch_size)
    weights = np.array([i for i in range(output_dim)])
    pdf_sum = np.array([0] * output_dim, dtype=np.float64)
    for i in predict_range[:-1]:
        predict_batch_data = [arr[i:i + predict_batch_size] for arr in X]
        pred_result = model.predict_on_batch(predict_batch_data)
        ctr_preds[i:i + predict_batch_size] = pred_result[0]
        batch_z_index = Z[i:i + predict_batch_size]
        row_index = np.arange(len(batch_z_index))
        pred_pdf = pred_result[1]
        Z_prob[i:i + predict_batch_size] = pred_pdf[row_index, batch_z_index]
        Z_pred[i:i + predict_batch_size] = pred_pdf.dot(weights)
        pdf_sum += pred_pdf.sum(axis=0)
    predict_batch_data = [arr[predict_range[-1]:predict_size] for arr in X]
    pred_result = model.predict_on_batch(predict_batch_data)
    ctr_preds[predict_range[-1]:predict_size] = pred_result[0]
    batch_z_index = Z[predict_range[-1]:predict_size]
    row_index = np.arange(len(batch_z_index))
    pred_pdf = pred_result[1]
    Z_prob[predict_range[-1]:predict_size] = pred_pdf[row_index, batch_z_index]
    Z_pred[predict_range[-1]:predict_size] = pred_pdf.dot(weights)
    pdf_sum += pred_pdf.sum(axis=0)
    pdf_avg = pdf_sum / predict_size

    return ctr_preds, Z_pred, Z_prob, pdf_avg