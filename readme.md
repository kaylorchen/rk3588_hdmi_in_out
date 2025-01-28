# 添加源
```bash
cat << 'EOF' | sudo tee /etc/apt/sources.list.d/kaylordut.list 
deb [signed-by=/etc/apt/keyrings/kaylor-keyring.gpg] http://apt.kaylordut.cn/kaylordut/ kaylordut main
EOF
sudo mkdir /etc/apt/keyrings -pv
sudo wget -O /etc/apt/keyrings/kaylor-keyring.gpg http://apt.kaylordut.cn/kaylor-keyring.gpg
sudo apt update
sudo apt install kaylordut-dev libbytetrack ai-instance
```



```bash
> v4l2-ctl -d /dev/video11 -V -D
Driver Info:
	Driver name      : rk_hdmirx
	Card type        : rk_hdmirx
	Bus info         : fdee0000.hdmirx-controller
	Driver version   : 5.10.160
	Capabilities     : 0x84201000
		Video Capture Multiplanar
		Streaming
		Extended Pix Format
		Device Capabilities
	Device Caps      : 0x04201000
		Video Capture Multiplanar
		Streaming
		Extended Pix Format
Format Video Capture Multiplanar:
	Width/Height      : 1920/1080
	Pixel Format      : 'BGR3' (24-bit BGR 8-8-8)
	Field             : None
	Number of planes  : 1
	Flags             : premultiplied-alpha, 0x000000fe
	Colorspace        : sRGB
	Transfer Function : Unknown (0x000000b8)
	YCbCr/HSV Encoding: Unknown (0x000000ff)
	Quantization      : Full Range
	Plane 0           :
	   Bytes per Line : 5760
	   Size Image     : 6220800
```