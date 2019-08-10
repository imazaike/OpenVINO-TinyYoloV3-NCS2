# OpenVINO-tinyYolov3
OpenVINO-tinyYolov3+ NCS2 + Raspberry pi3 model B+(raspbian stretch)/NUC(Ubuntu18.04)/DesktopPC(Ubuntu16.04) + USBcamera + Python3<br>
Inspired from https://github.com/PINTO0309/OpenVINO-YoloV3<br>
初心者のため、かなり手こずってしまったので、私と同じ右も左もわからない初心者向けにメモを残します。<br>
誠に勝手ながらPINTOさんの記事をかなり参考にさせていただきました。ありがとうございます。<br>
がんばって勉強中ですが、理解しきれていない所もあります。何か間違いなどお気づきになられましたら、是非とも教えていただきたいです。<br>

## tiny-YoloV3 + NCS2 SingleStick
```
$ python3 openvino_tiny-yolov3_MultiStick_test.py -numncs 1
```
## Environment
DesktopPC(Ubuntu16.04)<br>
NUC(Ubuntu18.04)<br>
Raspberry pi3 model B+(raspbian stretch armv7l)<br>

Python 3.6.9(DesktopPC)<br>
Python 3.6.8(NUC)<br>
Anaconda 4.7.11(DesktopPC)

## Environment construction procedure
### Work with DesktopPC(Ubuntu 16.04) & NUC(Ubuntu18.04)
Download the Intel® Distribution of OpenVINO™ toolkit package file from https://software.intel.com/en-us/openvino-toolkit/choose-download?elq_cid=5597297. Select the Intel® Distribution of OpenVINO™ toolkit for Linux package from the dropdown menu.<br>
If you have a previous version of the Intel Distribution of OpenVINO toolkit installed, rename or delete these two directories:<br>
```
/home/<user>/inference_engine_samples_build
/home/<user>/openvino_models
```

Open a command prompt terminal window and execute the following command.<br>
```
$ cd ~/Downloads/
$ tar -xvzf l_openvino_toolkit_p_<version>.tgz
$ rm l_openvino_toolkit_p_<version>.tgz
$ cd l_openvino_toolkit_p_<version>
```
install OpenVINO<br>
Please proceed without any changes<br>
```
sudo ./install_GUI.sh
```
To install the dependencies, execute the following command.<br>
```
$ cd /opt/intel/openvino/install_dependencies/
$ sudo -E ./install_openvino_dependencies.sh
$ nano ~/.bashrc
#Add this line to the end of the file:
source /opt/intel/openvino/bin/setupvars.sh

$ source ~/.bashrc
# Successful if displayed as below
[setupvars.sh] OpenVINO environment initialized

$ cd /opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites/
$ sudo ./install_prerequisites.sh
```
To enable NCS2, execute the following command.<br>
At this point, connect NCS2 to your PC.<br>
```
$ sudo usermod -a -G users "$(whoami)"
$ cat <<EOF > 97-usbboot.rules
SUBSYSTEM=="usb", ATTRS{idProduct}=="2150", ATTRS{idVendor}=="03e7", GROUP="users", MODE="0666", ENV{ID_MM_DEVICE_IGNORE}="1"
SUBSYSTEM=="usb", ATTRS{idProduct}=="2485", ATTRS{idVendor}=="03e7", GROUP="users", MODE="0666", ENV{ID_MM_DEVICE_IGNORE}="1"
SUBSYSTEM=="usb", ATTRS{idProduct}=="f63b", ATTRS{idVendor}=="03e7", GROUP="users", MODE="0666", ENV{ID_MM_DEVICE_IGNORE}="1"
EOF

$ sudo cp 97-usbboot.rules /etc/udev/rules.d/
$ sudo udevadm control --reload-rules
$ sudo udevadm trigger
$ sudo ldconfig
$ rm 97-usbboot.rules
```

### Work with RaspberryPi (Raspbian Stretch)
Download the Raspberry Pi version of openvino toolkit from the link above<br>
```
$ sudo apt update
$ sudo apt upgrade
$ cd ~/Downloads/
$ sudo mkdir -p /opt/intel/openvino
$ sudo tar -xf l_openvino_toolkit_raspbi_p_<version>.tgz --strip 1 -C /opt/intel/openvino
$ sudo apt install cmake
$ source /opt/intel/openvino/bin/setupvars.sh
$ nano ~/.bashrc
#Add this line to the end of the file:
source /opt/intel/openvino/bin/setupvars.sh

$ source ~/.bashrc
### Successful if displayed as below
[setupvars.sh] OpenVINO environment initialized

$ sudo usermod -a -G users "$(whoami)"
$ sudo reboot

$ sh /opt/intel/openvino/install_dependencies/install_NCS_udev_rules.sh
```

### Work with DesktopPC(Ubuntu 16.04)
Convert weight file(Darknet(.weights) -> Tensorflow(.pb) -> OpenVINO(.bin))<br>
1. Download this file<br>
https://drive.google.com/open?id=1n_njAPr3hfjWgHAyyaBrde4oF-3-W-W_<br>
2. Execute the following command<br>
```
$ cd ~
$ git clone https://github.com/PINTO0309/OpenVINO-YoloV3.git
$ cd OpenVINO-YoloV3.git
$ cp ~/Downloads/yolov3-tiny.weights weights
$ python3 convert_weights_pb.py \
--class_names coco.names \
--weights_file weights/yolov3-tiny.weights \
--data_format NHWC \
--tiny \
--output_graph pbmodels/frozen_yolov3-tiny-mine.pb
```
![weight-to-pb](https://user-images.githubusercontent.com/42289678/62821001-156ee600-bba8-11e9-8798-cd05571de65c.png)

```
$ sudo python3 /opt/intel//deployment_tools/model_optimizer/mo_tf.py \
--input_model pbmodels/frozen_yolov3-tiny-mine.pb \
--output_dir lrmodels/tiny-YoloV3/FP16 \
--data_type FP16 \
--batch 1 \
--tensorflow_use_custom_operations_config yolo_v3_tiny_changed.json
```



I fixed a little clone of Chuanqi305/MobileNetv2-SSDLite.<br>
https://github.com/chuanqi305/MobileNetv2-SSDLite.git<br>
I have not confirmed the behavior of the generated model at all.<br>
Pull Request is welcomed.<br>

### Environment
Ubuntu 16.04<br>
Python 2.7.12<br>
Python 3.5.2<br>

### Prerequisites
Tensorflow and Caffe version [SSD](https://github.com/weiliu89/caffe) is properly installed on your computer.

### Procedure

pip command and installation of Tensorflow.

```
$ cd ~
$ sudo apt-get install python-pip python-dev python-launchpadlib   # for Python 2.7
$ sudo apt-get install python3-pip python3-dev python3-launchpadlib # for Python 3.n
$ wget https://bootstrap.pypa.io/get-pip.py
$ sudo python3 get-pip.py
$ sudo python2 get-pip.py
$ sudo reboot
$ sudo -H pip install --upgrade tensorflow # Python 2.7; CPU support (no GPU support) Ver 1.8.0 [2018.06.03]
$ sudo -H pip3 install --upgrade tensorflow # Python 3.n; CPU support (no GPU support) Ver 1.8.0 [2018.06.03]
```

【Reference】Uninstall Tensorflow

```
$ sudo pip uninstall tensorflow  # for Python 2.7
$ sudo pip3 uninstall tensorflow # for Python 3.n
```

MobileNetv2-SSDLite Clone of learning data generation tool.

```
$ cd ~
$ git clone https://github.com/PINTO0309/MobileNetv2-SSDLite.git
$ cd ~/MobileNetv2-SSDLite/ssdlite
$ wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
$ tar -zxvf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
$ rm ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
```

Execute the following command.
