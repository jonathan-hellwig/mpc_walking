using StatsBase
using VegaLite, DataFrames
using Plots
using Distributions

using CoordinateTransformations
using Rotations
using StaticArrays

struct RobotDimensions
    torso_to_robot::SVector{3, Float64}
    robot_to_neck::SVector{3, Float64}
    robot_to_left_pelvis::SVector{3, Float64}
    robot_to_right_pelvis::SVector{3, Float64}
    hip_to_knee::SVector{3, Float64}
    knee_to_ankle::SVector{3, Float64}
    ankle_to_sole::SVector{3, Float64}
    robot_to_left_shoulder::SVector{3, Float64}
    robot_to_right_shoulder::SVector{3, Float64}
    left_shoulder_to_left_elbow::SVector{3, Float64}
    right_shoulder_to_right_elbow::SVector{3, Float64}
    elbow_to_wrist::SVector{3, Float64}
    neck_to_top_camera::SVector{3, Float64}
    neck_to_bottom_camera::SVector{3, Float64}
end

# Define the LegJoints struct
struct LegJoints
    hip_yaw_pitch::Float64
    hip_roll::Float64
    hip_pitch::Float64
    knee_pitch::Float64
    ankle_pitch::Float64
    ankle_roll::Float64
end

struct LegJointLimits
    hip_yaw_pitch::Tuple{Float64, Float64}
    hip_roll::Tuple{Float64, Float64}
    hip_pitch::Tuple{Float64, Float64}
    knee_pitch::Tuple{Float64, Float64}
    ankle_pitch::Tuple{Float64, Float64}
    ankle_roll::Tuple{Float64, Float64}
end

struct LegCollisionLimits
    pitch::Float64  # Pitch value corresponding to min_roll and max_roll
    min_roll::Float64   # Minimum roll value
    max_roll::Float64   # Maximum roll value
end

function left_leg_forward_kinematics(angles::LegJoints, dimensions::RobotDimensions)
    pelvis_to_robot = Translation(dimensions.robot_to_left_pelvis) ∘ LinearMap(RotX(π/4) * RotZ(-angles.hip_yaw_pitch) * RotX(-π/4))
    hip_to_pelvis = LinearMap(RotX(angles.hip_roll))
    thigh_to_hip = LinearMap(RotY(angles.hip_pitch))
    tibia_to_thigh = Translation(dimensions.hip_to_knee) ∘ LinearMap(RotY(angles.knee_pitch))
    ankle_to_tibia = Translation(dimensions.knee_to_ankle) ∘ LinearMap(RotY(angles.ankle_pitch))
    foot_to_ankle = LinearMap(RotX(angles.ankle_roll))
    sole_to_foot = Translation(dimensions.ankle_to_sole)
    return pelvis_to_robot ∘ hip_to_pelvis ∘ thigh_to_hip ∘ tibia_to_thigh ∘ ankle_to_tibia ∘ foot_to_ankle ∘ sole_to_foot
end

function interpolate_roll_limits(pitch_value, limits)
    for i in 1:length(limits) - 1
        if pitch_value >= limits[i].pitch && pitch_value <= limits[i + 1].pitch
            t = (pitch_value - limits[i].pitch) / (limits[i + 1].pitch - limits[i].pitch)
            min_roll = limits[i].min_roll + t * (limits[i + 1].min_roll - limits[i].min_roll)
            max_roll = limits[i].max_roll + t * (limits[i + 1].max_roll - limits[i].max_roll)
            return min_roll, max_roll
        end
    end
    error("Pitch value is outside the provided limits.")
end

# Forward pass of the kinematics using sampled joint angles
function sample_leg_joints_uniform(limits::LegJointLimits, collision_limits_leg::Array{LegCollisionLimits, 1})
    hip_yaw_pitch = rand(Uniform(limits.hip_yaw_pitch[1], limits.hip_yaw_pitch[2]))
    hip_roll = rand(Uniform(limits.hip_roll[1], limits.hip_roll[2]))
    hip_pitch = rand(Uniform(limits.hip_pitch[1], limits.hip_pitch[2]))
    knee_pitch = rand(Uniform(limits.knee_pitch[1], limits.knee_pitch[2]))
    ankle_pitch = rand(Uniform(limits.ankle_pitch[1], limits.ankle_pitch[2]))
    min_ankle_roll, max_ankle_roll = interpolate_roll_limits(ankle_pitch, collision_limits_leg)
    ankle_roll = rand(Uniform(min_ankle_roll, max_ankle_roll))
    
    return LegJoints(hip_yaw_pitch, hip_roll, hip_pitch, knee_pitch, ankle_pitch, ankle_roll)
end

# Define collision limits for the leg (LAnklePitch) in radians
const leg_collision_limits = [
    LegCollisionLimits(-1.189442, -0.049916, 0.075049),
    LegCollisionLimits(-0.840027, -0.179943, 0.169995),
    LegCollisionLimits(-0.700051, -0.397935, 0.220086),
    LegCollisionLimits(-0.449946, -0.397935, 0.768992),
    LegCollisionLimits(0.100007, -0.397935, 0.768992),
    LegCollisionLimits(0.349938, -0.397935, 0.550477),
    LegCollisionLimits(0.922755, 0.0, 0.049916)
]

# Define robot dimensions with the provided values
const robot_dimensions = RobotDimensions(
    (-0.0413, 0.0, -0.12842),                  # torso_to_robot
    (0.0, 0.0, 0.2115),                        # robot_to_neck
    (0.0, 0.05, 0.0),                          # robot_to_left_pelvis
    (0.0, -0.05, 0.0),                         # robot_to_right_pelvis
    (0.0, 0.0, -0.1),                          # hip_to_knee
    (0.0, 0.0, -0.1029),                       # knee_to_ankle
    (0.0, 0.0, -0.04519),                      # ankle_to_sole
    (0.0, 0.098, 0.185),                       # robot_to_left_shoulder
    (0.0, -0.098, 0.185),                      # robot_to_right_shoulder
    (0.105, 0.015, 0.0),                       # left_shoulder_to_left_elbow
    (0.105, -0.015, 0.0),                      # right_shoulder_to_right_elbow
    (0.05595, 0.0, 0.0),                       # elbow_to_wrist
    (0.05871, 0.0, 0.06364),                   # neck_to_top_camera
    (0.05071, 0.0, 0.01774)                    # neck_to_bottom_camera
)

const left_leg_limits = LegJointLimits(
    (-1.145303, 0.740810),   # LHipYawPitch limits
    (-0.379472, 0.790477),   # LHipRoll limits
    (-1.535889, 0.484090),   # LHipPitch limits
    (-0.092346, 2.112528),   # LKneePitch limits
    (-1.189442, 0.922755),   # LAnklePitch limits
    (-0.397880, 0.769001)    # LAnkleRoll limits
)

const number_of_samples = 10_000_000
const height_tolerance = 0.02
const roll_pitch_tolerance = 0.03
const number_of_yaw_bins = 10
const hip_height = -0.185

# Sample leg joints and compute forward kinematics
accepted_samples = []
for i in 1:number_of_samples
    sampled_leg_joints = sample_leg_joints_uniform(left_leg_limits, leg_collision_limits)
    left_foot_to_pelvis = left_leg_forward_kinematics(sampled_leg_joints, robot_dimensions)
    left_foot_in_pelvis = left_foot_to_pelvis(SVector(0.0, 0.0, 0.0))
    roll, pitch, yaw = Rotations.params(RotXYZ(left_foot_to_pelvis.linear))
    
    # Check if sample is valid
    if isapprox(left_foot_in_pelvis[3], hip_height, atol=height_tolerance) && 
        isapprox(roll, 0.0, atol=roll_pitch_tolerance) && 
        isapprox(pitch, 0.0, atol=roll_pitch_tolerance)
        push!(accepted_samples, (left_foot_in_pelvis[1:2], yaw))
    end
end

# Extract x, y, and yaw values from the accepted samples
x = [sample[1][1] for sample in accepted_samples]
y = [sample[1][2] for sample in accepted_samples]
yaws = [sample[2] for sample in accepted_samples]

# Create a histogram
yaw_histogram = fit(Histogram, yaws, nbins=number_of_yaw_bins)
yaws_bins = StatsBase.binindex.(Ref(yaw_histogram), yaws)

plot1 = histogram(yaws, bins=10)
# Subplots for each bin
df = DataFrame(x=x, y=y, z=rad2deg.(yaws), bin=yaws_bins)
plot2 = df |> @vlplot(:point, x=:x, y=:y, color={:z, scale={scheme=:plasma}}, columns=3, wrap=:bin)

# display(vcat(plot1, plot2))