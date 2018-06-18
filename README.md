# awesome-dl-docker
Awesome deep-learning projects packed in docker.

This is a curated set of docker files for awesome deep-learning
projects.

## Conventions

- Each sub-directory corresponds to a github project.  We usually
rename the github projects in our own system, when the project
name is not compatible with docker (containing capital letters) or
when it's too long.
- Each project should at least provide a Dockefile and a README.md
even if it's empty.
- The docker file should be considered a preservation of original
code and running environment.  The code should be directly cloned
from github, and should be checked out to a running commit.
Minimal changes should be made to the code, and when necessary,
changes should be implemented as a patch to the original code, and
the files needed for the patch should be checked into this git
repository.

## Dockerfile

## Naming

Examples:
- KittiSeg becomes kitti-seg.
- frustum-pointnet becomes f-pointnet.

