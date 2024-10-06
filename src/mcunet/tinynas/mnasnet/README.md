# Modified MnasNet

This subproject contains the search space implementation of the MnasNet paper
modified according to the contents of the MCUNet paper supplementary. In detail
that means:
1. The kernel size choices are extended by 7.
2. The layer choices are changed from multipliers to the fixed values
   [1, 2, 3, 4].
3. The expansion ratios for the mobile inverted bottleneck block is not based on
   the MobileNetV2 parameters but choosen from [3, 4, 6].

Also included are Pytorch modules implementeing the convolution operations.

The MnasNet paper can be found [here](https://openaccess.thecvf.com/content_CVPR_2019/html/Tan_MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile_CVPR_2019_paper).
