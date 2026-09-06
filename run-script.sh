git clone https://github.com/pothosware/SoapySDRPlay3.git ~/src/SoapySDRPlay3
cd ~/src/SoapySDRPlay3
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
sudo ldconfig

sudo systemctl status sdrplay.service --no-pager
SoapySDRUtil --find
