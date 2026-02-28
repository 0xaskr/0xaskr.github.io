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
            text: "Askr's Blog"
        },
        "site.description": {
            text: "A space for sharing my thoughts on technology, life, and personal projects."
        },
        "profile.description": {
            text: "Software engineer & Technology enthusiast"
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
            text: "Askr 的博客"
        },
        "site.description": {
            text: "分享关于技术、生活和个人项目的随笔空间。"
        },
        "profile.description": {
            text: "软件工程师 & 技术爱好者"
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
