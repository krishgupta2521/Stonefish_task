from setuptools import find_packages, setup

package_name = 'camera_rov'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='krish',
    maintainer_email='krishgupta2521@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'camera_node = camera_rov.camera_node:main',
            'object_detector = camera_rov.object_detector:main',
            'stonefish_object_detector = camera_rov.stonefish_object_detector:main',
        ],
    },
)
