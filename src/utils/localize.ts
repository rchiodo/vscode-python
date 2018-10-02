// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

import * as fs from 'fs';
import * as path from 'path';

// External callers of localize use these tables to retrieve localized values.
export namespace LanguageServiceSurveyBanner {
    export const bannerMessage = localize('LanguageServiceSurveyBanner.bannerMessage', 'Can you please take 2 minutes to tell us how the Python Language Server is working for you?');
    export const bannerLabelYes = localize('LanguageServiceSurveyBanner.bannerLabelYes', 'Yes, take survey now');
    export const bannerLabelNo = localize('LanguageServiceSurveyBanner.bannerLabelNo', 'No, thanks');
}

// Skip using vscode-nls and instead just compute our strings based on key values. Key values
// can be loaded out of the nls.<locale>.json files
let loadedCollection: { [index: string]: string } | undefined ;
let loadedLocale: string;

function localize(key: string, defValue: string) {
    // Return a pointer to function so that we refetch it on each call.
    return () => {
        return getString(key, defValue);
    };
}

function parseLocale() : string {
    // Attempt to load from the vscode locale. If not there, use english
    const vscodeConfigString = process.env.VSCODE_NLS_CONFIG;
    return vscodeConfigString ? JSON.parse(vscodeConfigString).locale : 'en-us';
}

function getString(key: string, defValue: string) {
    // Load the current collection
    if (!loadedCollection || parseLocale() !== loadedLocale) {
        loadedCollection = load();
    }

    // Lookup the string by the key
    if (loadedCollection && loadedCollection.hasOwnProperty(key)) {
        return loadedCollection[key];
    }

    // Not found, return the default
    return defValue;
}

function load() {
    // Figure out our current locale.
    loadedLocale = parseLocale();

    // Find the nls file that matches (if there is one)
    const nlsFile = path.join(__dirname, 'i18n', `localize.nls.${loadedLocale}.json`);
    if (fs.existsSync(nlsFile)) {
        const contents = fs.readFileSync(nlsFile, 'utf8');
        return JSON.parse(contents);
    }

    // If there isn't one, at least remember that we looked so we don't try to load a second time
    return {};
}

// Default to loading the current locale
load();
