from setuptools import setup, find_packages

setup(
    name='facenet_sim',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'opencv-python',
        'scikit-learn',
        'pillow',
        'tqdm'
    ],
    entry_points={
        'console_scripts': [
            'facenet-infer=facenet_sim.inference:main',
            'facenet-finetune=facenet_sim.finetune:main',
        ],
    },
)
