include({{ os.path.join(profile_dir, "cross-linux.profile") }})

[settings]
arch=x86_64

[conf]
tools.build:cflags+=["-march=haswell"]
tools.build:cxxflags+=["-march=haswell"]
