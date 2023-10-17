using CoordinateTransformations
using Distributions
using GLMakie
using Rotations
using StaticArrays

@kwdef struct RobotDimensions
    torso_to_robot::Any
    robot_to_neck::Any
    robot_to_left_pelvis::Any
    robot_to_right_pelvis::Any
    hip_to_knee::Any
    knee_to_ankle::Any
    ankle_to_sole::Any
    robot_to_left_shoulder::Any
    robot_to_right_shoulder::Any
    left_shoulder_to_left_elbow::Any
    right_shoulder_to_right_elbow::Any
    elbow_to_wrist::Any
    neck_to_top_camera::Any
    neck_to_bottom_camera::Any
end

NAO_DIMENSIONS = RobotDimensions(
    torso_to_robot = (-0.0413, 0.0, -0.12842),
    robot_to_neck = (0.0, 0.0, 0.2115),
    robot_to_left_pelvis = (0.0, 0.05, 0.0),
    robot_to_right_pelvis = (0.0, -0.05, 0.0),
    hip_to_knee = (0.0, 0.0, -0.1),
    knee_to_ankle = (0.0, 0.0, -0.1029),
    ankle_to_sole = (0.0, 0.0, -0.04519),
    robot_to_left_shoulder = (0.0, 0.098, 0.185),
    robot_to_right_shoulder = (0.0, -0.098, 0.185),
    left_shoulder_to_left_elbow = (0.105, 0.015, 0.0),
    right_shoulder_to_right_elbow = (0.105, -0.015, 0.0),
    elbow_to_wrist = (0.05595, 0.0, 0.0),
    neck_to_top_camera = (0.05871, 0.0, 0.06364),
    neck_to_bottom_camera = (0.05071, 0.0, 0.01774),
)

@kwdef struct LegJoints
    hip_yaw_pitch::Any
    hip_roll::Any
    hip_pitch::Any
    knee_pitch::Any
    ankle_pitch::Any
    ankle_roll::Any
end

function torso_to_robot(; torso_to_robot = NAO_DIMENSIONS.torso_to_robot)
    return Translation(-1.0 .* torso_to_robot)
end

function left_pelvis_to_robot(
    hip_yaw_pitch;
    robot_to_left_pelvis = NAO_DIMENSIONS.robot_to_left_pelvis,
)
    return Translation(robot_to_left_pelvis) ∘
           LinearMap(RotX(π / 4) * RotZ(-hip_yaw_pitch) * RotX(-π / 4))
end

function left_hip_to_left_pelvis(hip_roll)
    return LinearMap(RotX(hip_roll))
end

function left_thigh_to_left_hip(hip_pitch)
    return LinearMap(RotY(hip_pitch))
end

function left_tibia_to_left_thigh(knee_pitch; hip_to_knee = NAO_DIMENSIONS.hip_to_knee)
    return Translation(hip_to_knee) ∘ LinearMap(RotY(knee_pitch))
end

function left_ankle_to_left_tibia(ankle_pitch; knee_to_ankle = NAO_DIMENSIONS.knee_to_ankle)
    return Translation(knee_to_ankle) ∘ LinearMap(RotY(ankle_pitch))
end

function left_foot_to_left_ankle(ankle_roll; ankle_to_sole = NAO_DIMENSIONS.ankle_to_sole)
    return LinearMap(RotX(ankle_roll)) ∘ Translation(ankle_to_sole)
end

function left_foot_to_robot(leg_joints; dimensions = NAO_DIMENSIONS)
    return left_pelvis_to_robot(
        leg_joints.hip_yaw_pitch,
        robot_to_left_pelvis = dimensions.robot_to_left_pelvis,
    ) ∘ left_hip_to_left_pelvis(leg_joints.hip_roll) ∘
           left_thigh_to_left_hip(leg_joints.hip_pitch) ∘ left_tibia_to_left_thigh(
        leg_joints.knee_pitch,
        hip_to_knee = dimensions.hip_to_knee,
    ) ∘ left_ankle_to_left_tibia(
        leg_joints.ankle_pitch,
        knee_to_ankle = dimensions.knee_to_ankle,
    ) ∘ left_foot_to_left_ankle(
        leg_joints.ankle_roll,
        ankle_to_sole = dimensions.ankle_to_sole,
    )
end

function plot_robot(leg_joints)
    pelvis_to_robot = left_pelvis_to_robot(leg_joints.hip_yaw_pitch)
    hip_to_pelvis = left_hip_to_left_pelvis(leg_joints.hip_roll)
    thigh_to_hip = left_thigh_to_left_hip(leg_joints.hip_pitch)
    tibia_to_thigh = left_tibia_to_left_thigh(leg_joints.knee_pitch)
    ankle_to_tibia = left_ankle_to_left_tibia(leg_joints.ankle_pitch)
    foot_to_ankle = left_foot_to_left_ankle(leg_joints.ankle_roll)

    hip_to_robot = pelvis_to_robot ∘ hip_to_pelvis
    thigh_to_robot = hip_to_robot ∘ thigh_to_hip
    tibia_to_robot = thigh_to_robot ∘ tibia_to_thigh
    ankle_to_robot = tibia_to_robot ∘ ankle_to_tibia
    foot_to_robot = ankle_to_robot ∘ foot_to_ankle

    joint_positions = ([
        torso_to_robot()(SVector(0.0, 0.0, 0.0)),
        SVector(0.0, 0.0, 0.0),
        pelvis_to_robot(SVector(0.0, 0.0, 0.0)),
        hip_to_robot(SVector(0.0, 0.0, 0.0)),
        thigh_to_robot(SVector(0.0, 0.0, 0.0)),
        tibia_to_robot(SVector(0.0, 0.0, 0.0)),
        ankle_to_robot(SVector(0.0, 0.0, 0.0)),
        foot_to_robot(SVector(0.0, 0.0, 0.0)),
    ])
    figure = Figure()
    axis = Axis3(figure[1, 1], aspect = (1, 1, 1))
    lines!(axis, joint_positions)
    scatter!(axis, joint_positions)
    figure
end

# const LEFT_LEG_LIMITS = LegJoints(
#     hip_yaw_pitch = -1.145303:0.740810,
#     hip_roll = -0.379472:0.790477,
#     hip_pitch = -1.535889:0.484090,
#     knee_pitch = -0.092346:2.112528,
#     ankle_pitch = -1.189442:0.922755,
#     ankle_roll = -0.397880:0.769001,
# )
#
# struct LegCollisionLimits
#     pitch::Any
#     min_roll::Any
#     max_roll::Any
# end
#
#
# function interpolate_roll_limits(pitch_value, limits)
#     for i = 1:length(limits)-1
#         if pitch_value >= limits[i].pitch && pitch_value <= limits[i+1].pitch
#             t = (pitch_value - limits[i].pitch) / (limits[i+1].pitch - limits[i].pitch)
#             min_roll = limits[i].min_roll + t * (limits[i+1].min_roll - limits[i].min_roll)
#             max_roll = limits[i].max_roll + t * (limits[i+1].max_roll - limits[i].max_roll)
#             return min_roll, max_roll
#         end
#     end
#     error("Pitch value is outside the provided limits.")
# end
#
# # Forward pass of the kinematics using sampled joint angles
# function sample_leg_joints_uniform(
#     limits::LegJointLimits,
#     collision_limits_leg::Array{LegCollisionLimits,1},
# )
#     hip_yaw_pitch = rand(Uniform(limits.hip_yaw_pitch[1], limits.hip_yaw_pitch[2]))
#     hip_roll = rand(Uniform(limits.hip_roll[1], limits.hip_roll[2]))
#     hip_pitch = rand(Uniform(limits.hip_pitch[1], limits.hip_pitch[2]))
#     knee_pitch = rand(Uniform(limits.knee_pitch[1], limits.knee_pitch[2]))
#     ankle_pitch = rand(Uniform(limits.ankle_pitch[1], limits.ankle_pitch[2]))
#     min_ankle_roll, max_ankle_roll =
#         interpolate_roll_limits(ankle_pitch, collision_limits_leg)
#     ankle_roll = rand(Uniform(min_ankle_roll, max_ankle_roll))
#
#     return LegJoints(
#         hip_yaw_pitch,
#         hip_roll,
#         hip_pitch,
#         knee_pitch,
#         ankle_pitch,
#         ankle_roll,
#     )
# end
#
# # Define collision limits for the leg (LAnklePitch) in radians
# const leg_collision_limits = [
#     LegCollisionLimits(-1.189442, -0.049916, 0.075049),
#     LegCollisionLimits(-0.840027, -0.179943, 0.169995),
#     LegCollisionLimits(-0.700051, -0.397935, 0.220086),
#     LegCollisionLimits(-0.449946, -0.397935, 0.768992),
#     LegCollisionLimits(0.100007, -0.397935, 0.768992),
#     LegCollisionLimits(0.349938, -0.397935, 0.550477),
#     LegCollisionLimits(0.922755, 0.0, 0.049916),
# ]
#
# # Define robot dimensions with the provided values
#
# const left_leg_limits = LegJointLimits(
#     (-1.145303, 0.740810),   # LHipYawPitch limits
#     (-0.379472, 0.790477),   # LHipRoll limits
#     (-1.535889, 0.484090),   # LHipPitch limits
#     (-0.092346, 2.112528),   # LKneePitch limits
#     (-1.189442, 0.922755),   # LAnklePitch limits
#     (-0.397880, 0.769001),    # LAnkleRoll limits
# )
#
# const number_of_samples = 10_000_000
# const height_tolerance = 0.02
# const roll_pitch_tolerance = 0.03
# const number_of_yaw_bins = 10
# const hip_height = -0.185
#
# # Sample leg joints and compute forward kinematics
# accepted_samples = []
# for i = 1:number_of_samples
#     sampled_leg_joints = sample_leg_joints_uniform(left_leg_limits, leg_collision_limits)
#     left_foot_to_pelvis = left_leg_forward_kinematics(sampled_leg_joints, robot_dimensions)
#     left_foot_in_pelvis = left_foot_to_pelvis(SVector(0.0, 0.0, 0.0))
#     roll, pitch, yaw = Rotations.params(RotXYZ(left_foot_to_pelvis.linear))
#
#     # Check if sample is valid
#     if isapprox(left_foot_in_pelvis[3], hip_height, atol = height_tolerance) &&
#        isapprox(roll, 0.0, atol = roll_pitch_tolerance) &&
#        isapprox(pitch, 0.0, atol = roll_pitch_tolerance)
#         push!(accepted_samples, (left_foot_in_pelvis[1:2], yaw))
#     end
# end
#
# # Extract x, y, and yaw values from the accepted samples
# x = [sample[1][1] for sample in accepted_samples]
# y = [sample[1][2] for sample in accepted_samples]
# yaws = [sample[2] for sample in accepted_samples]
#
# # # Create a histogram
# # yaw_histogram = fit(Histogram, yaws, nbins=number_of_yaw_bins)
# # yaws_bins = StatsBase.binindex.(Ref(yaw_histogram), yaws)
# #
# # plot1 = histogram(yaws, bins=10)
# # # Subplots for each bin
# # df = DataFrame(x=x, y=y, z=rad2deg.(yaws), bin=yaws_bins)
# # plot2 = df |> @vlplot(:point, x = :x, y = :y, color = {:z, scale = {scheme = :plasma}}, columns = 3, wrap = :bin)
#
# # display(vcat(plot1, plot2))
