# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/unitree/Downloads/unitree_legged_sdk

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/unitree/Downloads/unitree_legged_sdk/build

# Include any dependencies generated for this target.
include CMakeFiles/lcm_position.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/lcm_position.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/lcm_position.dir/flags.make

CMakeFiles/lcm_position.dir/examples/lcm_position.cpp.o: CMakeFiles/lcm_position.dir/flags.make
CMakeFiles/lcm_position.dir/examples/lcm_position.cpp.o: ../examples/lcm_position.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/unitree/Downloads/unitree_legged_sdk/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/lcm_position.dir/examples/lcm_position.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/lcm_position.dir/examples/lcm_position.cpp.o -c /home/unitree/Downloads/unitree_legged_sdk/examples/lcm_position.cpp

CMakeFiles/lcm_position.dir/examples/lcm_position.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lcm_position.dir/examples/lcm_position.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/unitree/Downloads/unitree_legged_sdk/examples/lcm_position.cpp > CMakeFiles/lcm_position.dir/examples/lcm_position.cpp.i

CMakeFiles/lcm_position.dir/examples/lcm_position.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lcm_position.dir/examples/lcm_position.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/unitree/Downloads/unitree_legged_sdk/examples/lcm_position.cpp -o CMakeFiles/lcm_position.dir/examples/lcm_position.cpp.s

CMakeFiles/lcm_position.dir/examples/lcm_position.cpp.o.requires:

.PHONY : CMakeFiles/lcm_position.dir/examples/lcm_position.cpp.o.requires

CMakeFiles/lcm_position.dir/examples/lcm_position.cpp.o.provides: CMakeFiles/lcm_position.dir/examples/lcm_position.cpp.o.requires
	$(MAKE) -f CMakeFiles/lcm_position.dir/build.make CMakeFiles/lcm_position.dir/examples/lcm_position.cpp.o.provides.build
.PHONY : CMakeFiles/lcm_position.dir/examples/lcm_position.cpp.o.provides

CMakeFiles/lcm_position.dir/examples/lcm_position.cpp.o.provides.build: CMakeFiles/lcm_position.dir/examples/lcm_position.cpp.o


# Object files for target lcm_position
lcm_position_OBJECTS = \
"CMakeFiles/lcm_position.dir/examples/lcm_position.cpp.o"

# External object files for target lcm_position
lcm_position_EXTERNAL_OBJECTS =

lcm_position: CMakeFiles/lcm_position.dir/examples/lcm_position.cpp.o
lcm_position: CMakeFiles/lcm_position.dir/build.make
lcm_position: CMakeFiles/lcm_position.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/unitree/Downloads/unitree_legged_sdk/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable lcm_position"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lcm_position.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/lcm_position.dir/build: lcm_position

.PHONY : CMakeFiles/lcm_position.dir/build

CMakeFiles/lcm_position.dir/requires: CMakeFiles/lcm_position.dir/examples/lcm_position.cpp.o.requires

.PHONY : CMakeFiles/lcm_position.dir/requires

CMakeFiles/lcm_position.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/lcm_position.dir/cmake_clean.cmake
.PHONY : CMakeFiles/lcm_position.dir/clean

CMakeFiles/lcm_position.dir/depend:
	cd /home/unitree/Downloads/unitree_legged_sdk/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/unitree/Downloads/unitree_legged_sdk /home/unitree/Downloads/unitree_legged_sdk /home/unitree/Downloads/unitree_legged_sdk/build /home/unitree/Downloads/unitree_legged_sdk/build /home/unitree/Downloads/unitree_legged_sdk/build/CMakeFiles/lcm_position.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/lcm_position.dir/depend

