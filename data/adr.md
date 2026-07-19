# ADR — Autonomous Disinfectant Robot for Hospitals (Mechatronics NUST project)

The Autonomous Disinfectant Robot (ADR) is a final-year design project of the Department of Mechatronics Engineering, NUST College of EME (2026), on display at the department's stall at Indus RAS Expo 2026. ADR is an autonomous mobile robot that disinfects indoor hospital environments with three cleaning modes at once — floor mopping, vacuum cleaning, and air fumigation spray — navigating on its own using LiDAR-based SLAM. It was built by Muhammad Farzam Amir and Muhammad Arham Amir, supervised by Dr Ayesha Zeb, Dr Zohaib Riaz and Dr Kanwal Naveed.

> Source: ADR final-year project thesis, NUST CEME, 2026

## Why the ADR disinfection robot was built

Hospital-acquired infections (HAIs) affect 7–10% of hospitalized patients in developed countries, adding an average of 4–5 extra hospital days, with an even greater burden in low- and middle-income countries. Manual cleaning is labor-intensive and inconsistent, and repeatedly exposes cleaning staff to harmful chemicals. Existing commercial hospital robots are expensive and mostly limited to a single disinfection mode (UV-C), and lack the maneuverability needed in narrow corridors. ADR aims to be an affordable, autonomous, multi-mode alternative that protects both patients and healthcare workers.

## How the ADR robot works — mecanum wheels, LiDAR SLAM, dual controllers

ADR drives on four mecanum wheels, giving fully omnidirectional movement — it can translate sideways, rotate in place, and move diagonally without turning first, ideal for tight, furniture-dense hospital rooms and corridors. A 360-degree RPLIDAR scanner feeds LiDAR-based SLAM (slam_toolbox with AMCL on ROS) running on an NVIDIA Jetson Nano, which handles mapping, localization and coverage path planning. An ESP32 microcontroller does real-time closed-loop PID motor control with encoder feedback and switches the disinfection modules, communicating with the Jetson over a serial link with a safety watchdog that halts the robot if commands stop. Power comes from a 16 V, 18 Ah lithium-ion pack (plus a dedicated battery for the Jetson), giving about 1.6 hours of continuous operation. An operator can start and monitor disinfection cycles remotely over Wi-Fi.

## ADR disinfection system — mop, vacuum and fumigation

ADR carries three independent disinfection modules: a floor mopping unit with a pump and solenoid-controlled disinfectant flow, a vacuum suction unit for loose debris, and an air fumigation spray that disperses fine droplets to treat elevated surfaces and airborne particles. Together they address floor, surface and aerial contamination in a single pass — something single-modality UV-C robots and consumer cleaning robots do not offer.

## ADR test results and hospital deployment

In testing across simulated hospital layouts, ADR consistently covered over 80% of the floor area with reliable static and dynamic obstacle avoidance. The mopping module removed the majority of visible contaminants, the vacuum collected fine particles and lightweight debris, and the fumigation spray achieved good vertical reach and air dispersion. The robot was also deployed in a real hospital room with a bed, IV stand, seats and tables, where it mapped the room and cleaned autonomously. Compared with commercial UV-C disinfection robots, ADR offers multi-modal disinfection at roughly a hundredth of the cost. Future work includes adding a UV-C LED array as a fourth mode, microbiological validation, and multi-robot coordination.
