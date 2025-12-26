include({{ os.path.join(profile_dir, "cross-linux.profile") }})

[settings]
# armv6hf doesn't exist
arch=armv6

[conf]
tools.build:cflags+=["-march=armv6", "-mfpu=vfp", "-mfloat-abi=hard"]
tools.build:cxxflags+=["-march=armv6", "-mfpu=vfp", "-mfloat-abi=hard"]
tools.build:exelinkflags+=["-latomic"]
tools.build:sharedlinkflags+=["-latomic"]
