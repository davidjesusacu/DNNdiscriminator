# DNNdiscriminator
Requires:
 Theano
 DCGAN (https://github.com/Newmu/dcgan_code)
 
 
# Setup
Add the path of the DCGAN repo in settings.py DCGAN_ROOT= 


# Example
python discriminator.py test/

Input images will be cropped to 227x227 to remove irrelevant details and captions