import React, { useEffect, useRef } from "react";
import { View, Text, StyleSheet, Animated } from "react-native";

interface ProgressBarProps {
  progress: number; // 0–1 float (matches how App.tsx calls it: downloadProgress / 100)
}

const ProgressBar: React.FC<ProgressBarProps> = ({ progress }) => {
  const animatedWidth = useRef(new Animated.Value(0)).current;
  const clampedProgress = Math.max(0, Math.min(progress, 1));
  const displayPct = Math.round(clampedProgress * 100);

  useEffect(() => {
    Animated.timing(animatedWidth, {
      toValue: clampedProgress,
      duration: 300,
      useNativeDriver: false,
    }).start();
  }, [clampedProgress]);

  // Color shifts green as it completes
  const barColor = animatedWidth.interpolate({
    inputRange: [0, 0.5, 1],
    outputRange: ['#3B82F6', '#6366F1', '#22C55E'],
  });

  return (
    <View style={styles.wrapper}>
      {/* Label row */}
      <View style={styles.labelRow}>
        <Text style={styles.labelLeft}>Downloading model…</Text>
        <Text style={styles.labelRight}>{displayPct}%</Text>
      </View>

      {/* Track */}
      <View style={styles.track}>
        {/* Animated fill */}
        <Animated.View
          style={[
            styles.fill,
            {
              width: animatedWidth.interpolate({
                inputRange: [0, 1],
                outputRange: ['0%', '100%'],
              }),
              backgroundColor: barColor,
            },
          ]}
        />

        {/* Shimmer stripe overlay */}
        <View style={styles.shimmerOverlay} />
      </View>

      {/* Sub-label */}
      <Text style={styles.subLabel}>
        {displayPct < 100 ? 'Please keep the app open' : '✓ Download complete!'}
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  wrapper: {
    paddingHorizontal: 4,
    paddingVertical: 8,
  },

  labelRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  labelLeft: {
    fontSize: 13,
    fontWeight: '600',
    color: '#1E293B',
    letterSpacing: 0.2,
  },
  labelRight: {
    fontSize: 18,
    fontWeight: '800',
    color: '#3B82F6',
    letterSpacing: -0.5,
  },

  track: {
    height: 12,
    backgroundColor: '#E2E8F0',
    borderRadius: 999,
    overflow: 'hidden',
    position: 'relative',
  },
  fill: {
    height: '100%',
    borderRadius: 999,
  },
  shimmerOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    borderRadius: 999,
    // Subtle top-highlight for depth
    borderTopWidth: 1,
    borderTopColor: 'rgba(255,255,255,0.4)',
  },

  subLabel: {
    fontSize: 11,
    color: '#94A3B8',
    marginTop: 6,
    textAlign: 'center',
    letterSpacing: 0.1,
  },
});

export default ProgressBar;