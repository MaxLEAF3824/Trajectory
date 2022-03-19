/*
 Navicat Premium Data Transfer

 Source Server         : mysql56
 Source Server Type    : MySQL
 Source Server Version : 50651
 Source Host           : localhost:3306
 Source Schema         : trajectory

 Target Server Type    : MySQL
 Target Server Version : 50651
 File Encoding         : 65001

 Date: 19/03/2022 14:16:45
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for trajectory
-- ----------------------------
DROP TABLE IF EXISTS `trajectory`;
CREATE TABLE `trajectory`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `length` int(11) NOT NULL,
  `points` text CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Compact;

SET FOREIGN_KEY_CHECKS = 1;
