The folder holds the original slice data set, with a size of 512*512, and is organized as follows:

  +train                      *Training set
    +images                *Raw data, .tif format
      +1-C11             
      +2-C12_real
      +3-C12_imag
      +4-C22
      +5-alpha
      +6-anisotropy
      +7-entropy
      +8-windspeed
      +9-rainfall
    +labels_0-1            *Data tag with two pixel values of 0 and 1 inside the tag in .png format
    +labels_0-255        *Data label visualization with two pixel values of 0 and 255 inside the label in .png format

 +test                        *Test set
    +images                *Raw data, .tif format
      +1-C11             
      +2-C12_real
      +3-C12_imag
      +4-C22
      +5-alpha
      +6-anisotropy
      +7-entropy
      +8-windspeed
      +9-rainfall
    +labels_0-1            *Data tag with two pixel values of 0 and 1 inside the tag in .png format
    +labels_0-255        *Data label visualization with two pixel values of 0 and 255 inside the label in .png format

+eval                         *Eval set
    +images                *Raw data, .tif format
      +1-C11             
      +2-C12_real
      +3-C12_imag
      +4-C22
      +5-alpha
      +6-anisotropy
      +7-entropy
      +8-windspeed
      +9-rainfall
    +labels_0-1            *Data tag with two pixel values of 0 and 1 inside the tag in .png format
    +labels_0-255        *Data label visualization with two pixel values of 0 and 255 inside the label in .png format