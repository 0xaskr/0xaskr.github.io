/**
 * This configures the translations for all ui text in your website. 
 * 
 * All languages will follow this ordering/structure and will fallback to the
 * default language for any entries that haven't been translated 
 */
import type { SupportedLanguage } from "src/utils/i18n";

export default {
    "en": {
        "site.title": {
            text: "Astro Theme Cody"
        },
        "site.description": {
            text: "A minimalist blog theme built with Astro. A quick and easy starter build for anyone who wants to start their own blog."
        },
        "profile.description": {
            text: "your bio description"
        },
        "blog.lastUpdated": {
            text: "Last updated:"
        },
        "sidebar.tableOfContents": {
            text: "Table of Contents"
        },
        "project.platform": {
            text: "PLATFORM"
        },
        "project.stack": {
            text: "STACK"
        },
        "project.website": {
            text: "WEBSITE"
        }
    },
    "zh": {
        "site.title": {
            text: "Astro Theme Cody"
        },
        "site.description": {
            text: "一个基于 Astro 构建的极简博客主题。为你快速搭建个人博客提供简单的入门。"
        },
        "profile.description": {
            text: "你的个人简介"
        },
        "blog.lastUpdated": {
            text: "最后更新："
        },
        "sidebar.tableOfContents": {
            text: "目录"
        },
        "project.platform": {
            text: "平台"
        },
        "project.stack": {
            text: "技术栈"
        },
        "project.website": {
            text: "网站"
        }
    }
} as const satisfies TranslationUIEntries;

type TranslationUIEntries = Record<SupportedLanguage, Record<string, UIEntry>>;

export type UIEntry = { text: string };