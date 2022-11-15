import io
import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


logger = logging.getLogger('tensorflow')
epsilon = sys.float_info.epsilon


def count_pdf(Z, z_size):
    total_num = len(Z)
    z_count = [0 for _ in range(z_size)]
    z_pdf = [0 for _ in range(z_size)]
    for z in Z:
        z_count[int(z)] += 1
    for z in range(z_size):
        z_pdf[z] = z_count[z] / total_num + epsilon
    return z_pdf


def count_cdf(Z, z_size):
    z_cdf = [0 for _ in range(z_size)]
    z_pdf = count_pdf(Z, z_size)
    prev_z_cdf = 0
    for z in range(z_size):
        z_cdf[z] = z_pdf[z] + prev_z_cdf
        prev_z_cdf = z_cdf[z]
    return z_cdf


def pdf2cdf(z_pdf):
    z_size = len(z_pdf)
    z_cdf = [0 for _ in range(z_size)]
    prev_z_cdf = 0
    for z in range(z_size):
        z_cdf[z] = z_pdf[z] + prev_z_cdf
        prev_z_cdf = z_cdf[z]
    return z_cdf


def cdf2pdf(z_cdf):
    z_size = len(z_cdf)
    z_pdf = [0 for _ in range(z_size)]
    z_pdf[0] = z_cdf[0] + epsilon
    for z in range(1, z_size):
        z_pdf[z] = z_cdf[z] - z_cdf[z-1] + epsilon
    return z_pdf


def avg_pdf(z_pdf):
    return np.mean(z_pdf, axis=0)


def avg_cdf(z_pdf):
    avg_z_pdf = avg_pdf(z_pdf)
    z_cdf = pdf2cdf(avg_z_pdf)
    return z_cdf


def count_plot_pdf(Z, z_size, label=''):
    z_list = [z for z in range(z_size)]
    pdf = count_pdf(Z, z_size)
    plt.plot(z_list, pdf, label=label)


def count_plot_cdf(Z, z_size, label=''):
    z_list = [z for z in range(z_size)]
    cdf = count_cdf(Z, z_size)
    plt.plot(z_list, cdf, label=label)


def plot_mp_result_regression(Z, Z_pred, predict_z_size=301, save_fig=False, show_fig=True, figure_path='', title=''):
    # pdf figure
    figure = plt.figure()
    count_plot_pdf(Z, predict_z_size, 'ground truth')
    count_plot_pdf(Z_pred, predict_z_size, 'predict')
    plt.title('{} pdf'.format(title))
    plt.legend()
    if save_fig:
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        plt.savefig(figure_path + '{}_pdf.pdf'.format('_'.join(title.split())), bbox_inches='tight')
    if show_fig:
        plt.show()

    # cdf figure
    plt.figure()
    count_plot_cdf(Z, predict_z_size, 'ground truth')
    count_plot_cdf(Z_pred, predict_z_size, 'predict')
    plt.title('{} cdf'.format(title))
    plt.legend()
    if save_fig:
        plt.savefig(figure_path + '{}_cdf.pdf'.format('_'.join(title.split())), bbox_inches='tight')
    if show_fig:
        plt.show()
    return figure


def plot_mp_result(z_truth, z_pred, save_fig=False, show_fig=True, figure_path='', title='', tf_summary=True, epoch=0):
    z_list = [z for z in range(len(z_truth))]
    figure = plt.figure()
    plt.plot(z_list, z_truth, label='ground truth')
    plt.plot(z_list, z_pred, label='predict')
    plt.title('{}'.format(title))
    plt.legend()
    if save_fig:
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        plt.savefig(figure_path + '{}.pdf'.format('_'.join(title.split())), bbox_inches='tight')
    if show_fig:
        plt.show()
    if tf_summary:
        result_img = plot_to_image(figure)
        tf.summary.image(title, result_img, step=epoch)
    return figure


def plot_mp_case(z_truth, z_pred, save_fig=False, show_fig=True, figure_path='', title='', tf_summary=True, epoch=0):
    z_list = [z for z in range(len(z_pred))]
    figure = plt.figure()
    plt.title('{}'.format(title))
    plt.axvline(x=z_truth, lw=2, ls="-", c="red", label='z={}'.format(z_truth))
    plt.plot(z_list, z_pred, label='predict')
    plt.legend()
    if save_fig:
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        plt.savefig(figure_path + '{}.pdf'.format('_'.join(title.split())), bbox_inches='tight')
    if show_fig:
        plt.show()
    if tf_summary:
        result_img = plot_to_image(figure)
        tf.summary.image(title, result_img, step=epoch)
    return figure


def plot_to_image(figure):
  """
  Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call.
  """
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image






