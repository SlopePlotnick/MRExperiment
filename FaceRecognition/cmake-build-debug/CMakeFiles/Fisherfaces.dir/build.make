# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/plotnickslope/Desktop/学习资料/模式识别/作业/FaceRecognition

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/plotnickslope/Desktop/学习资料/模式识别/作业/FaceRecognition/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/Fisherfaces.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/Fisherfaces.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/Fisherfaces.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Fisherfaces.dir/flags.make

CMakeFiles/Fisherfaces.dir/Fisherfaces.cpp.o: CMakeFiles/Fisherfaces.dir/flags.make
CMakeFiles/Fisherfaces.dir/Fisherfaces.cpp.o: /Users/plotnickslope/Desktop/学习资料/模式识别/作业/FaceRecognition/Fisherfaces.cpp
CMakeFiles/Fisherfaces.dir/Fisherfaces.cpp.o: CMakeFiles/Fisherfaces.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/plotnickslope/Desktop/学习资料/模式识别/作业/FaceRecognition/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Fisherfaces.dir/Fisherfaces.cpp.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Fisherfaces.dir/Fisherfaces.cpp.o -MF CMakeFiles/Fisherfaces.dir/Fisherfaces.cpp.o.d -o CMakeFiles/Fisherfaces.dir/Fisherfaces.cpp.o -c /Users/plotnickslope/Desktop/学习资料/模式识别/作业/FaceRecognition/Fisherfaces.cpp

CMakeFiles/Fisherfaces.dir/Fisherfaces.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Fisherfaces.dir/Fisherfaces.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/plotnickslope/Desktop/学习资料/模式识别/作业/FaceRecognition/Fisherfaces.cpp > CMakeFiles/Fisherfaces.dir/Fisherfaces.cpp.i

CMakeFiles/Fisherfaces.dir/Fisherfaces.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Fisherfaces.dir/Fisherfaces.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/plotnickslope/Desktop/学习资料/模式识别/作业/FaceRecognition/Fisherfaces.cpp -o CMakeFiles/Fisherfaces.dir/Fisherfaces.cpp.s

# Object files for target Fisherfaces
Fisherfaces_OBJECTS = \
"CMakeFiles/Fisherfaces.dir/Fisherfaces.cpp.o"

# External object files for target Fisherfaces
Fisherfaces_EXTERNAL_OBJECTS =

Fisherfaces: CMakeFiles/Fisherfaces.dir/Fisherfaces.cpp.o
Fisherfaces: CMakeFiles/Fisherfaces.dir/build.make
Fisherfaces: /opt/homebrew/lib/libopencv_gapi.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_stitching.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_alphamat.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_aruco.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_bgsegm.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_bioinspired.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_ccalib.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_dnn_objdetect.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_dnn_superres.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_dpm.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_face.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_freetype.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_fuzzy.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_hfs.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_img_hash.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_intensity_transform.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_line_descriptor.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_mcc.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_quality.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_rapid.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_reg.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_rgbd.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_saliency.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_sfm.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_stereo.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_structured_light.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_superres.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_surface_matching.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_tracking.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_videostab.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_viz.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_wechat_qrcode.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_xfeatures2d.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_xobjdetect.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_xphoto.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_shape.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_highgui.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_datasets.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_plot.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_text.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_ml.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_phase_unwrapping.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_optflow.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_ximgproc.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_video.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_videoio.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_imgcodecs.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_objdetect.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_calib3d.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_dnn.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_features2d.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_flann.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_photo.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_imgproc.4.8.1.dylib
Fisherfaces: /opt/homebrew/lib/libopencv_core.4.8.1.dylib
Fisherfaces: CMakeFiles/Fisherfaces.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/plotnickslope/Desktop/学习资料/模式识别/作业/FaceRecognition/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Fisherfaces"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Fisherfaces.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Fisherfaces.dir/build: Fisherfaces
.PHONY : CMakeFiles/Fisherfaces.dir/build

CMakeFiles/Fisherfaces.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Fisherfaces.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Fisherfaces.dir/clean

CMakeFiles/Fisherfaces.dir/depend:
	cd /Users/plotnickslope/Desktop/学习资料/模式识别/作业/FaceRecognition/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/plotnickslope/Desktop/学习资料/模式识别/作业/FaceRecognition /Users/plotnickslope/Desktop/学习资料/模式识别/作业/FaceRecognition /Users/plotnickslope/Desktop/学习资料/模式识别/作业/FaceRecognition/cmake-build-debug /Users/plotnickslope/Desktop/学习资料/模式识别/作业/FaceRecognition/cmake-build-debug /Users/plotnickslope/Desktop/学习资料/模式识别/作业/FaceRecognition/cmake-build-debug/CMakeFiles/Fisherfaces.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Fisherfaces.dir/depend

