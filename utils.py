"""
Some codes from https://github.com/Newmu/dcgan_code
"""
#Modified line 23 to read fits file

from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange
from astropy.io import fits
import os as os

from astropy.io import fits
hdulist = fits.open('SRC22_NVSS.FITS')
img = hdulist[0].data
header = hdulist[0].header
hdulist.close()
header['CRPIX1'] = 75.5
header['CRPIX2'] = 75.5

"""load_path = '/Users/A2Z/Desktop/DCGAN-tensorflow-master/data/nvss/'
file_list = os.listdir(load_path)
len_file_list = len(file_list)"""

pp = pprint.PrettyPrinter()


get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])
n = 300 #128
m = 300 #128

def get_image(image_path, input_height, input_width,
              resize_height=n, resize_width=m,
              is_crop=True, is_grayscale=False):

  #image = imread(image_path, is_grayscale)
  #for file in file_list:
  hdulist = fits.open(image_path) # Modified to read fits image
  img = hdulist[0].data
  image = img/np.max(img)
  hdulist.close()
#  resize_height,resize_width = shape(img)[0]
#  print (resize_height,resize_width)

  return transform(image, input_height, input_width,
                   resize_height, resize_width, is_crop)

def save_fits_image(image, image_path):

  if image.shape[-1]==1:
    for i in range(3):
      flat_image=image.reshape(image.shape[0:-1])
  else:
    flat_image=image

  header1 = header[0:15]  
  hdu = fits.PrimaryHDU(flat_image, header=header1)
  return hdu.writeto(image_path,overwrite=True)


def save_single_image(image, image_path):
  return imsave_single(inverse_transform(image), image_path)

def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
  if (is_grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
  return inverse_transform(images)

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  print('merge: image h,w',h,w)
  img = np.zeros((h * size[0], w * size[1], 3))
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image
  return img

def imsave(images, size, path):
  return scipy.misc.imsave(path, merge(images, size))

def imsave_single(image, path):
  image_rgb = np.zeros((image.shape[0], image.shape[1], 3))
  if len(image.shape)==2:
    for i in range(3):
      image_rgb[:,:,i]=image
  elif image.shape[-1]==1:
    for i in range(3):
      image_rgb[:,:,i]=image[:,:,0]
  else:
    image_rgb=image
    
  return scipy.misc.imsave(path, image_rgb)

def center_crop(x, crop_h, crop_w,
                resize_h=n, resize_w=m):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, 
              resize_height=n, resize_width=m, is_crop=True):
  if is_crop:
    cropped_image = center_crop(
      image, input_height, input_width, 
      resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
  return (images+1.)/2.

def to_json(output_path, *layers):
  with open(output_path, "w") as layer_f:
    lines = ""
    for w, b, bn in layers:
      layer_idx = w.name.split('/')[0].split('h')[1]

      B = b.eval()

      if "lin/" in w.name:
        W = w.eval()
        depth = W.shape[1]
      else:
        W = np.rollaxis(w.eval(), 2, 0)
        depth = W.shape[0]

      biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
      if bn != None:
        gamma = bn.gamma.eval()
        beta = bn.beta.eval()

        gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
        beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
      else:
        gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
        beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

      if "lin/" in w.name:
        fs = []
        for w in W.T:
          fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

        lines += """
          var layer_%s = {
            "layer_type": "fc", 
            "sy": 1, "sx": 1, 
            "out_sx": 1, "out_sy": 1,
            "stride": 1, "pad": 0,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
      else:
        fs = []
        for w_ in W:
          fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

        lines += """
          var layer_%s = {
            "layer_type": "deconv", 
            "sy": 5, "sx": 5,
            "out_sx": %s, "out_sy": %s,
            "stride": 2, "pad": 1,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
               W.shape[0], W.shape[3], biases, gamma, beta, fs)
    layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)

def visualize(sess, dcgan, config, option,nplt=100,out_dir='./samples/'):

  print('Producing simulations in the folder: ',out_dir)
  
  image_frame_dim = int(math.ceil(config.batch_size**.5))
  #if nplt is None: nplt=100
  if option == 0:
    #zspace will be generated with Uniform sampling
    print('Option 0: z uniform png')
    z_sample = np.random.uniform(0, 1, size=(config.batch_size, dcgan.z_dim))
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, [image_frame_dim, image_frame_dim],
                os.path.join(out_dir,'/test_zuniform_%s.png' % strftime("%Y-%m-%d-%H", gmtime())) )
    
  elif option == 1:
    #zspace is one hot sequential
    
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(nplt):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      if config.dataset == "mnist":
        y = np.random.choice(10, config.batch_size)
        y_one_hot = np.zeros((config.batch_size, 10))
        y_one_hot[np.arange(config.batch_size), y] = 1

        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
      else:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

      save_images(samples, [image_frame_dim, image_frame_dim], os.path.join(out_dir,'test_arange_%s.png' % (idx)) )
      
  elif option == 2:
    #zspace on-hot random gif
    
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in [random.randint(0, 99) for _ in xrange(nplt)]:
      print("Option 2: one-hot random gif - %d" % idx)
      z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
      z_sample = np.tile(z, (config.batch_size, 1))
      #z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      if config.dataset == "mnist":
        y = np.random.choice(10, config.batch_size)
        y_one_hot = np.zeros((config.batch_size, 10))
        y_one_hot[np.arange(config.batch_size), y] = 1

        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
      else:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

      try:
        make_gif(samples, os.path.join(out_dir,'test_gif_%s.gif' % (idx)) )
      except:
        save_images(samples, [image_frame_dim, image_frame_dim], os.path.join(out_dir,'test_%s.png' % strftime("%Y-%m-%d %H:%M:%S", gmtime())) )
        
  elif option == 3:
    #zspace one-hot sequential gif
    
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(nplt):
      print("Option 3: z one-hot gif - %d" % idx)        
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, os.path.join(out_dir,'test_gif_%s.gif' % (idx)) )
      
  elif option == 4:
    #zspace one-hot sequential gif merged - animation
    
    image_set = []
    values = np.arange(0, 1, 1./config.batch_size)

    for idx in xrange(nplt):
      print("Option 4: z one-hot gif anim - %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

      image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
      make_gif(image_set[-1], os.path.join(out_dir,'test_gif_%s.gif' % (idx)) )

    new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
                     for idx in range(n) + range((n-1), -1, -1)]
    make_gif(new_image_set, os.path.join(out_dir,'test_gif_merged.gif'), duration=8)

  if option == 5:
    #zspace will be generated with Uniform sampling
    
    image_set = []
    for idx in xrange(nplt):
      print("option 5: zuniform to gif anim -  %d" % idx)
      
      z_sample = np.random.uniform(-1, 1, size=(config.batch_size, dcgan.z_dim))
      image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))

      print('images_set len,element.shape: ',len(image_set),image_set[-1].shape)
      for iimg,img in enumerate(image_set[-1]):
        fname=os.path.join(out_dir,'test_zuni_%s_%s.png' % (idx,iimg) )
        print('saving image (shape, fname) ',img[:,:,0].shape,fname)
        save_single_image(img,fname)
        #make_gif(img[:,:,0], fname)

    #new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
    #                 for idx in range(n) + range((n-1), -1, -1)]
    #make_gif(new_image_set, os.path.join(out_dir,'test_zuni_gif_merged.gif' ), duration=8)
    
  if option == 6:
    #zspace will be generated with Uniform sampling output: FITS image 

    for idx in xrange(nplt):
      print("option 5: zuniform to gif anim -  %d" % idx)
      
      z_sample = np.random.uniform(-1, 1, size=(config.batch_size, dcgan.z_dim))
      image_batch=sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

      for iimg,img in enumerate(image_batch):
        fname=os.path.join(out_dir,'test_zuni_batch%s_%s.fits' % (idx,iimg) ) 
        print('saving image (shape, fname) ',img[:,:,0].shape,fname)
        save_fits_image(img,fname)
