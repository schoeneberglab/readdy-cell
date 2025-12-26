include({{ os.path.join(profile_dir, "cross-linux.profile") }})

[settings]
arch=armv8

[conf]
tools.build:cflags+=["-mcpu=cortex-a53+crc+simd"]
tools.build:cxxflags+=["-mcpu=cortex-a53+crc+simd"]
