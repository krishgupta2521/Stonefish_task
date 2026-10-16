#pragma once
// Published KYUBIC overall envelope, in metres.
namespace kyubic_geometry {
constexpr double LENGTH = 0.570;
constexpr double WIDTH  = 0.550;
constexpr double HEIGHT = 0.522;
// First-pass mounting offsets; verify against the real robot before tuning.
constexpr double HORIZONTAL_X = 0.19;
constexpr double HORIZONTAL_Y = 0.14;
constexpr double VERTICAL_Y = 0.16;
constexpr double VERTICAL_Z = 0.08;
}
