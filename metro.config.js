const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');

const defaultConfig = getDefaultConfig(__dirname);

/**
 * Metro configuration
 * https://reactnative.dev/docs/metro
 *
 * @type {import('@react-native/metro-config').MetroConfig}
 */
const config = {
  resolver: {
    assetExts: Array.from(new Set([
      ...defaultConfig.resolver.assetExts,
      'onnx',
      'bin',
      'txt',
      'json',
      'gguf',
    ])),
  },
};

module.exports = mergeConfig(defaultConfig, config);
