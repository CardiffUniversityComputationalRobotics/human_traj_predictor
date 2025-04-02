import os
from glob import glob
from setuptools import find_packages, setup

package_name = "human_traj_predictor"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # ! PKL
        (
            os.path.join(
                "share",
                package_name,
                "human_traj_predictor",
                "ped_pred",
                "TrainedModel",
                "uncertainty_aware_model",
                "CollisionGrid",
            ),
            glob(
                os.path.join(
                    "human_traj_predictor/ped_pred/TrainedModel/uncertainty_aware_model/CollisionGrid",
                    "*.pkl",
                )
            ),
        ),
        # ! CHECKPOINT
        (
            os.path.join(
                "share",
                package_name,
                "human_traj_predictor",
                "ped_pred",
                "TrainedModel",
                "uncertainty_aware_model",
                "CollisionGrid",
            ),
            glob(
                os.path.join(
                    "human_traj_predictor/ped_pred/TrainedModel/uncertainty_aware_model/CollisionGrid",
                    "*.tar",
                )
            ),
        ),
        # ! CHECKPOINT
        (
            os.path.join(
                "share",
                package_name,
                "human_traj_predictor",
                "ped_pred",
                "Data",
                "HBS",
            ),
            glob(
                os.path.join(
                    "human_traj_predictor/ped_pred/Data/HBS",
                    "*.csv",
                )
            ),
        ),
        # ! CHECKPOINT
        (
            os.path.join(
                "share",
                package_name,
                "human_traj_predictor",
                "ped_pred",
                "Data",
                "HBS",
            ),
            glob(
                os.path.join(
                    "human_traj_predictor/ped_pred/Data/HBS",
                    "*.csv",
                )
            ),
        ),
        # ! CPKL
        (
            os.path.join(
                "share",
                package_name,
                "human_traj_predictor",
                "ped_pred",
                "Data",
                "preprocessed",
            ),
            glob(
                os.path.join(
                    "human_traj_predictor/ped_pred/Data/preprocessed",
                    "*.cpkl",
                )
            ),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="sasm",
    maintainer_email="sasilva1998@gmail.com",
    description="TODO: Package description",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "human_traj_predictor_node = human_traj_predictor.human_traj_predictor:main"
        ],
    },
)
