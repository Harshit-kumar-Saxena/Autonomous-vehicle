"""
Unit tests for robot model kinematics.
"""

import pytest
import math
import sys
from pathlib import Path

# Add aethernav_stack parent to path for imports
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from aethernav_stack.core.types import Pose2D, Twist2D, WheelVelocities
from aethernav_stack.robot_model.kinematics import DiffDriveKinematics


class TestDiffDriveKinematics:
    """Tests for differential drive kinematics."""
    
    @pytest.fixture
    def kinematics(self):
        """Create a kinematics instance with test parameters."""
        return DiffDriveKinematics(
            wheel_radius=0.05,  # 5cm
            track_width=0.20    # 20cm
        )
    
    def test_inverse_straight_forward(self, kinematics):
        """Test inverse kinematics for straight forward motion."""
        twist = Twist2D(linear=0.5, angular=0.0)
        wheels = kinematics.inverse(twist)
        
        # Both wheels should have same velocity
        assert abs(wheels.left - wheels.right) < 1e-6
        
        # Wheel velocity should be v / r
        expected_omega = 0.5 / 0.05  # 10 rad/s
        assert abs(wheels.left - expected_omega) < 1e-6
    
    def test_inverse_rotation_in_place(self, kinematics):
        """Test inverse kinematics for rotation in place."""
        twist = Twist2D(linear=0.0, angular=1.0)  # 1 rad/s rotation
        wheels = kinematics.inverse(twist)
        
        # Wheels should spin opposite directions
        assert wheels.left * wheels.right < 0
        
        # |left| should equal |right|
        assert abs(abs(wheels.left) - abs(wheels.right)) < 1e-6
    
    def test_forward_from_inverse(self, kinematics):
        """Test that forward(inverse(twist)) == twist."""
        original = Twist2D(linear=0.3, angular=0.5)
        
        wheels = kinematics.inverse(original)
        recovered = kinematics.forward(wheels)
        
        assert abs(recovered.linear - original.linear) < 1e-6
        assert abs(recovered.angular - original.angular) < 1e-6
    
    def test_pose_delta_straight(self, kinematics):
        """Test pose delta for straight motion."""
        # Both wheels at 10 rad/s for 1 second
        wheels = WheelVelocities(left=10.0, right=10.0)
        
        delta = kinematics.compute_pose_delta(wheels, dt=1.0, current_theta=0.0)
        
        # Should move forward by v*t = (omega * r) * t = 10 * 0.05 * 1 = 0.5m
        assert abs(delta.x - 0.5) < 1e-6
        assert abs(delta.y) < 1e-6
        assert abs(delta.theta) < 1e-6
    
    def test_pose_delta_arc(self, kinematics):
        """Test pose delta for arc motion."""
        # Different wheel speeds for turning
        wheels = WheelVelocities(left=8.0, right=12.0)
        
        delta = kinematics.compute_pose_delta(wheels, dt=0.1, current_theta=0.0)
        
        # Should have some forward motion and rotation
        assert delta.x > 0
        assert delta.theta != 0
    
    def test_clamp_preserves_ratio(self, kinematics):
        """Test that clamping preserves wheel velocity ratio."""
        wheels = WheelVelocities(left=20.0, right=10.0)  # 2:1 ratio
        
        clamped = kinematics.clamp_wheel_velocities(
            wheels, 
            max_wheel_vel=15.0,
            preserve_ratio=True
        )
        
        # Ratio should be preserved
        original_ratio = wheels.left / wheels.right
        clamped_ratio = clamped.left / clamped.right
        assert abs(original_ratio - clamped_ratio) < 1e-6
        
        # Max should be at limit
        assert clamped.left == 15.0


class TestPose2D:
    """Tests for Pose2D type."""
    
    def test_addition(self):
        """Test pose addition."""
        p1 = Pose2D(x=1.0, y=2.0, theta=0.5)
        p2 = Pose2D(x=0.5, y=0.5, theta=0.1)
        
        result = p1 + p2
        
        assert abs(result.x - 1.5) < 1e-6
        assert abs(result.y - 2.5) < 1e-6
        assert abs(result.theta - 0.6) < 1e-6
    
    def test_angle_normalization(self):
        """Test that angles are normalized to [-pi, pi]."""
        p1 = Pose2D(theta=math.pi)
        p2 = Pose2D(theta=0.1)
        
        result = p1 + p2
        
        # Should wrap around
        assert -math.pi <= result.theta <= math.pi


class TestWheelVelocities:
    """Tests for WheelVelocities type."""
    
    def test_to_pwm(self):
        """Test conversion to PWM."""
        wheels = WheelVelocities(left=5.0, right=10.0)
        
        left_pwm, right_pwm = wheels.to_pwm(max_wheel_vel=10.0)
        
        assert left_pwm == 127  # 50% of 255
        assert right_pwm == 255  # 100% of 255
    
    def test_get_directions(self):
        """Test direction detection."""
        forward = WheelVelocities(left=5.0, right=5.0)
        backward = WheelVelocities(left=-5.0, right=-5.0)
        
        assert forward.get_directions() == (1, 1)
        assert backward.get_directions() == (0, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
