/**
 * This configures the navigation bar and footer. Each entry is a nav link with
 * the correct translation and url path.
 * 
 * All languages will follow this ordering/structure and will fallback to the
 * default language for any entries that haven't been translated 
 */

import type { SupportedLanguage } from "src/utils/i18n";

export default {
    "en": {
        "home": {
            text: "Home",
            slug: ""
        },
        "about": {
            text: "About",
            slug: "about"
        },
        "blog": {
            text: "Blog",
            slug: "blog",
            route: "/blog/pages/1"
        },
        "projects": {
            text: "Projects",
            slug: "projects",
            route: "/projects/pages/1"
        },
        "archive": {
            text: "Archive",
            slug: "archive"
        },
        "tags": {
            text: "Tags",
            slug: "tags"
        },
        "series": {
            text: "Series",
            slug: "series"
        }
    },
    "zh": {
        "home": {
            text: "首页",
            slug: ""
        },
        "about": {
            text: "关于",
            slug: "about"
        },
        "blog": {
            text: "博客",
            slug: "blog",
            route: "/blog/pages/1"
        },
        "projects": {
            text: "项目",
            slug: "projects",
            route: "/projects/pages/1"
        },
        "archive": {
            text: "归档",
            slug: "archive"
        },
        "tags": {
            text: "标签",
            slug: "tags"
        },
        "series": {
            text: "系列",
            slug: "series"
        }
    }
} as const satisfies TranslationNavEntries;

type TranslationNavEntries = Record<SupportedLanguage, Record<string, NavEntry>>

export type NavEntry = {
    /*
        Provided translation
    */
    text: string,

    /*
        Content collection slug or url path for this page without the language code
    */
    slug: string,

    route?: string
};