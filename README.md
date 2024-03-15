# SOD-diffusion: Diffusion Model for Salient Object Detection

## Datasets
All datasets are available in public.
* Download the DUTS-TR and DUTS-TE from [Here](http://saliencydetection.net/duts/#org3aad434)
* Download the DUT-OMRON from [Here](http://saliencydetection.net/dut-omron/#org96c3bab)
* Download the HKU-IS from [Here](https://i.cs.hku.hk/~yzyu/research/deep_saliency.html)
* Download the ECSSD from [Here](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)
* Download the PASCAL-S from [Here](http://cbs.ic.gatech.edu/salobj/)

## Requirements
* Python >= 3.8.x
* Pytorch >= 2.0.1
* diffusers >= 0.25.1
* pip install -r requirements.txt

## Run
* Run **train.sh** and **inference.sh** scripts.
<pre><code># For training SOD-diffusion
cd scripts
bash train.sh

# For inference SOD-diffusion  
cd scripts
bash inference.sh
</code></pre>

## Configurations
--pretrained_model_name_or_path: pretrained model path from hugging face or local dir.  
--swinb_model_path: pretrained swinb model.  
--train_img_list: img_list.txt, including the absolute path of all train images.  
--train_gt_list: gt_list.txt, including the absolute path all ground truth masks.  
--val_img: path of the validation set of images.  
--val_gt: path of the validation set of ground truth masks.

