# Update
sudo apt update

# Install Compiler
# Versions missing from the distro archive (e.g. clang-18 on jammy) come from apt.llvm.org.
sudo apt install -y $COMPILER || {
  curl -fsSL https://apt.llvm.org/llvm.sh | sudo bash -s -- "${COMPILER#*-}"
  sudo apt install -y $COMPILER
}
$COMPILER -v
echo "CC=$COMPILER" >>$GITHUB_ENV
COMPILER_NAME=${COMPILER%-*}
COMPILER_VERSION=${COMPILER#*-}
if [[ $COMPILER_NAME == "clang" ]]; then
  echo "CXX=$COMPILER_NAME++-$COMPILER_VERSION" >>$GITHUB_ENV
else
  echo "CXX=g++-$COMPILER_VERSION" >>$GITHUB_ENV
fi

# Install MKL
wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install intel-oneapi-compiler-dpcpp-cpp intel-oneapi-mkl-devel intel-oneapi-mkl intel-oneapi-openmp

#source /opt/intel/oneapi/mkl/latest/env/vars.sh
source /opt/intel/oneapi/setvars.sh
#source /opt/intel/oneapi/latest/oneapi-vars.sh


echo "MKLROOT=$MKLROOT" >>$GITHUB_ENV
echo "INTEL=/opt/intel/oneapi" >>$GITHUB_ENV
echo "ONEAPI_ROOT=$ONEAPI_ROOT" >>$GITHUB_ENV

# Install other dependencies
sudo apt install cmake
